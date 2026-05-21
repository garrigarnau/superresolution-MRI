"""
Benchmark script for MRI Super-Resolution.
Runs Swin2SR and Real-ESRGAN inference on test LR images,
saves super-resolved outputs and timing data.
"""

import time
import json
import argparse
import torch
import numpy as np
from PIL import Image

from config import (
    REAL_ESRGAN_RESULTS_DIR,
    REAL_ESRGAN_WEIGHTS_PATH,
    RESULTS_DIR,
    SWIN2SR_RESULTS_DIR,
    TEST_LR_DIR,
)
from swin2sr_alignment import align_to_lr_footprint


def resolve_device(device_name):
    """Resolve and validate the requested inference device."""
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif device_name == "gpu":
        device_name = "mps" if torch.backends.mps.is_available() else "cuda"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or use --device auto/cpu."
        )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is false. "
            "Use --device auto/cpu on this machine."
        )

    return torch.device(device_name)


def run_swin2sr(lr_dir, output_dir, device):
    """Run the canonical aligned Swin2SR baseline on all test images."""
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

    print(f"Loading Swin2SR model on {device}...")
    processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
    model = model.to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    lr_images = sorted(lr_dir.glob("*.png"))

    print(f"Running Swin2SR on {len(lr_images)} images...")
    for i, img_path in enumerate(lr_images):
        # Load grayscale and convert to RGB (model expects 3 channels)
        lr_gray = Image.open(img_path).convert("L")
        img = lr_gray.convert("RGB")

        # Preprocess
        inputs = processor(img, return_tensors="pt").to(device)

        # Inference with timing
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.time() - start
        timings.append(elapsed)

        # Postprocess
        output = outputs.reconstruction.squeeze().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        # output shape: (3, H, W) → take mean across channels for grayscale
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))  # (H, W, 3)
            output = np.mean(output, axis=2).astype(np.uint8)

        sr_img = Image.fromarray(output, mode="L")

        # Ensure output is 256x256
        if sr_img.size != (256, 256):
            sr_img = sr_img.resize((256, 256), Image.BICUBIC)

        aligned_output = align_to_lr_footprint(np.array(sr_img), lr_gray)
        Image.fromarray(aligned_output, mode="L").save(output_dir / img_path.name)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(lr_images)}] avg time: {np.mean(timings):.4f}s")

    return timings


def run_real_esrgan(lr_dir, output_dir, device):
    """Run Real-ESRGAN x4 on all test images."""
    from RealESRGAN import RealESRGAN

    print(f"Loading Real-ESRGAN model on {device}...")
    model = RealESRGAN(device, scale=4)
    model.load_weights(REAL_ESRGAN_WEIGHTS_PATH.as_posix(), download=True)
    predict = model.predict
    if device.type == "cuda" and hasattr(model.predict, "__wrapped__"):
        # The ai-forever package decorates predict() with CUDA autocast.
        # On some GPUs this returns NaNs for these grayscale MRI inputs.
        predict = lambda img: model.predict.__wrapped__(model, img)

    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    lr_images = sorted(lr_dir.glob("*.png"))

    print(f"Running Real-ESRGAN on {len(lr_images)} images...")
    for i, img_path in enumerate(lr_images):
        # Load as RGB (required by Real-ESRGAN)
        img = Image.open(img_path).convert("RGB")

        # Inference with timing
        start = time.time()
        sr_image = predict(img)
        elapsed = time.time() - start
        timings.append(elapsed)

        # Convert back to grayscale
        sr_image = sr_image.convert("L")

        # Ensure output is 256x256
        if sr_image.size != (256, 256):
            sr_image = sr_image.resize((256, 256), Image.BICUBIC)

        sr_image.save(output_dir / img_path.name)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(lr_images)}] avg time: {np.mean(timings):.4f}s")

    return timings


def main():
    parser = argparse.ArgumentParser(description="Run MRI super-resolution benchmarks.")
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cuda", "mps", "cpu"],
        default="auto",
        help="Inference device. 'gpu' picks MPS when available, otherwise CUDA.",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Test images: {TEST_LR_DIR}")
    print()

    # Run the canonical aligned Swin2SR baseline
    swin2sr_timings = run_swin2sr(TEST_LR_DIR, SWIN2SR_RESULTS_DIR, device)
    print(f"\nSwin2SR done: {len(swin2sr_timings)} images, "
          f"avg {np.mean(swin2sr_timings):.4f}s/image\n")

    # Run Real-ESRGAN
    esrgan_timings = run_real_esrgan(TEST_LR_DIR, REAL_ESRGAN_RESULTS_DIR, device)
    print(f"\nReal-ESRGAN done: {len(esrgan_timings)} images, "
          f"avg {np.mean(esrgan_timings):.4f}s/image\n")

    # Save timings
    timings_data = {
        "device": str(device),
        "n_images": len(swin2sr_timings),
        "swin2sr": {
            "mean": float(np.mean(swin2sr_timings)),
            "std": float(np.std(swin2sr_timings)),
            "total": float(np.sum(swin2sr_timings)),
            "per_image": swin2sr_timings,
        },
        "real_esrgan": {
            "mean": float(np.mean(esrgan_timings)),
            "std": float(np.std(esrgan_timings)),
            "total": float(np.sum(esrgan_timings)),
            "per_image": esrgan_timings,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "timings.json", "w") as f:
        json.dump(timings_data, f, indent=2)

    print("Timings saved to results/timings.json")


if __name__ == "__main__":
    main()
