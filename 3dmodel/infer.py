"""
Inference for 3D MRI Super-Resolution.
Uses sliding window over full LR volumes, stitches patches into SR volume.
Also generates trilinear baseline for comparison.
"""

import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import zoom

from config import (
    CHECKPOINT_DIR,
    ENCODER_NAME,
    INFER_PATCH_OVERLAP,
    INFER_PATCH_SIZE,
    RESULTS_DIR,
    PATCHES_DIR,
    SCALE_FACTOR,
)
from model import MedicalNetSR3D


def sliding_window_inference(model, lr_volume, patch_size, overlap, device):
    """
    Run inference on a full LR volume using sliding window.

    Args:
        model: trained SR model
        lr_volume: (D, H, W) numpy array, normalized [0, 1]
        patch_size: size of cubic input patches
        overlap: overlap between patches
        device: torch device

    Returns:
        sr_volume: (D*2, H*2, W*2) reconstructed SR volume
    """
    model.eval()
    d, h, w = lr_volume.shape
    stride = patch_size - overlap

    # Output volume (x2)
    out_d, out_h, out_w = d * SCALE_FACTOR, h * SCALE_FACTOR, w * SCALE_FACTOR
    sr_volume = np.zeros((out_d, out_h, out_w), dtype=np.float32)
    weight_map = np.zeros((out_d, out_h, out_w), dtype=np.float32)

    # Pad LR volume to fit patches evenly
    pad_d = (stride - (d - patch_size) % stride) % stride
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride

    lr_padded = np.pad(lr_volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')

    with torch.no_grad():
        for di in range(0, lr_padded.shape[0] - patch_size + 1, stride):
            for hi in range(0, lr_padded.shape[1] - patch_size + 1, stride):
                for wi in range(0, lr_padded.shape[2] - patch_size + 1, stride):
                    # Extract LR patch
                    lr_patch = lr_padded[
                        di:di + patch_size,
                        hi:hi + patch_size,
                        wi:wi + patch_size,
                    ]

                    # Model inference
                    lr_tensor = torch.from_numpy(lr_patch).float().unsqueeze(0).unsqueeze(0).to(device)
                    sr_patch = model(lr_tensor).squeeze().cpu().numpy()

                    # Place in output volume (x2 coordinates)
                    out_ps = patch_size * SCALE_FACTOR
                    od, oh, ow = di * SCALE_FACTOR, hi * SCALE_FACTOR, wi * SCALE_FACTOR

                    # Only write to valid region (not padding)
                    valid_d = min(out_ps, out_d - od)
                    valid_h = min(out_ps, out_h - oh)
                    valid_w = min(out_ps, out_w - ow)

                    if valid_d > 0 and valid_h > 0 and valid_w > 0:
                        sr_volume[od:od + valid_d, oh:oh + valid_h, ow:ow + valid_w] += \
                            sr_patch[:valid_d, :valid_h, :valid_w]
                        weight_map[od:od + valid_d, oh:oh + valid_h, ow:ow + valid_w] += 1.0

    # Average overlapping regions
    weight_map = np.maximum(weight_map, 1e-8)
    sr_volume /= weight_map

    return np.clip(sr_volume, 0, 1)


def trilinear_baseline(lr_volume, scale_factor=SCALE_FACTOR):
    """Simple trilinear upsampling as baseline."""
    return zoom(lr_volume, scale_factor, order=1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 3D SR inference on test volumes.")
    parser.add_argument("--checkpoint", type=str,
                        default=str(CHECKPOINT_DIR / "best_model.pth"),
                        help="Path to decoder checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Device
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model...")
    model = MedicalNetSR3D(ENCODER_NAME).to(device)

    # Load trained decoder weights
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, PSNR: {checkpoint['best_psnr']:.2f} dB)")

    model.eval()

    # Output directories
    sr_dir = RESULTS_DIR / "model_sr"
    trilinear_dir = RESULTS_DIR / "trilinear"
    sr_dir.mkdir(parents=True, exist_ok=True)
    trilinear_dir.mkdir(parents=True, exist_ok=True)

    # Process test volumes
    test_volumes_dir = PATCHES_DIR / "test_volumes"
    lr_files = sorted(test_volumes_dir.glob("*_lr.npy"))

    print(f"\nProcessing {len(lr_files)} test volumes...")
    for lr_path in lr_files:
        subj_id = lr_path.stem.replace("_lr", "")
        print(f"  {subj_id}...")

        lr_volume = np.load(lr_path)

        # Model SR (sliding window)
        sr_volume = sliding_window_inference(
            model, lr_volume, INFER_PATCH_SIZE, INFER_PATCH_OVERLAP, device
        )
        np.save(sr_dir / f"{subj_id}_sr.npy", sr_volume)

        # Trilinear baseline
        trilinear_vol = trilinear_baseline(lr_volume)
        np.save(trilinear_dir / f"{subj_id}_trilinear.npy", trilinear_vol)

        print(f"    SR shape: {sr_volume.shape}, Trilinear shape: {trilinear_vol.shape}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  Model SR: {sr_dir}")
    print(f"  Trilinear baseline: {trilinear_dir}")


if __name__ == "__main__":
    main()
