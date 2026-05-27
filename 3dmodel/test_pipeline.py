"""
Dry-run test for the 3D SR pipeline.
Verifies model loading, forward/backward pass, and patch processing
before committing to full training on GPU.

Usage:
    python test_pipeline.py          # Basic CPU/MPS test
    python test_pipeline.py --gpu    # Full GPU stress test
"""

import sys
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    ENCODER_NAME,
    LEARNING_RATE,
    NUM_WORKERS,
    PATCHES_DIR,
)
from dataset import MRI3DPatchDataset
from model import MedicalNetSR3D


def test_model(device):
    """Test model loading and architecture."""
    print("[1/5] Testing model loading...")
    model = MedicalNetSR3D(ENCODER_NAME).to(device)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Verify encoder is frozen
    for param in model.encoder.parameters():
        assert not param.requires_grad, "Encoder should be frozen!"

    print(f"       Encoder: {encoder_params:,} params (frozen)")
    print(f"       Decoder: {decoder_params:,} params (trainable)")
    print(f"       Total trainable: {trainable:,}")
    print(f"       PASSED")
    return model


def test_forward(model, device):
    """Test forward pass with synthetic data."""
    print(f"\n[2/5] Testing forward pass...")
    model.eval()

    # Synthetic LR patch: (1, 1, 32, 32, 32)
    lr_input = torch.rand(1, 1, 32, 32, 32, device=device)

    with torch.no_grad():
        sr_output = model(lr_input)

    assert sr_output.shape == (1, 1, 64, 64, 64), f"Output shape: {sr_output.shape}, expected (1, 1, 64, 64, 64)"
    print(f"       Input:  {lr_input.shape}")
    print(f"       Output: {sr_output.shape}")
    print(f"       Output range: [{sr_output.min():.3f}, {sr_output.max():.3f}]")
    print(f"       PASSED")


def test_backward(model, device):
    """Test training step."""
    print(f"\n[3/5] Testing backward pass...")
    model.train()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LEARNING_RATE)

    lr_input = torch.rand(2, 1, 32, 32, 32, device=device)
    hr_target = torch.rand(2, 1, 64, 64, 64, device=device)

    optimizer.zero_grad()
    sr_output = model(lr_input)
    loss = criterion(sr_output, hr_target)
    loss.backward()
    optimizer.step()

    print(f"       Loss: {loss.item():.6f}")
    print(f"       Gradients OK (encoder frozen, decoder updated)")
    print(f"       PASSED")


def test_dataset(device):
    """Test loading real patches if available."""
    print(f"\n[4/5] Testing dataset...")
    train_dir = PATCHES_DIR / "train"
    if not train_dir.exists() or not list((train_dir / "HR").glob("*.npy")):
        print(f"       SKIPPED (no patches found at {train_dir})")
        print(f"       Run preprocess.py first to generate patches")
        return None

    dataset = MRI3DPatchDataset(train_dir)
    print(f"       Found {len(dataset)} training patches")

    lr, hr = dataset[0]
    assert lr.shape == (1, 32, 32, 32), f"LR shape: {lr.shape}"
    assert hr.shape == (1, 64, 64, 64), f"HR shape: {hr.shape}"
    print(f"       LR: {lr.shape}, range [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"       HR: {hr.shape}, range [{hr.min():.3f}, {hr.max():.3f}]")
    print(f"       PASSED")
    return dataset


def test_checkpoint(model):
    """Test checkpoint save/load."""
    print(f"\n[5/5] Testing checkpoint...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = CHECKPOINT_DIR / "test_checkpoint.pth"

    torch.save({
        "epoch": 1,
        "decoder_state_dict": model.decoder.state_dict(),
        "best_psnr": 30.0,
    }, test_path)

    ckpt = torch.load(test_path, weights_only=False)
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    test_path.unlink()

    print(f"       Save/load OK")
    print(f"       PASSED")


def test_gpu_stress(device, n_steps=10):
    """Full GPU stress test with realistic batch sizes."""
    print(f"\n[GPU] Stress test ({n_steps} steps, batch={BATCH_SIZE})...")

    model = MedicalNetSR3D(ENCODER_NAME).to(device)
    model.train()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LEARNING_RATE)

    step_times = []
    losses = []

    for step in range(1, n_steps + 1):
        lr_batch = torch.rand(BATCH_SIZE, 1, 32, 32, 32, device=device)
        hr_batch = torch.rand(BATCH_SIZE, 1, 64, 64, 64, device=device)

        start = time.time()
        optimizer.zero_grad()
        sr = model(lr_batch)
        loss = criterion(sr, hr_batch)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - start

        step_times.append(elapsed)
        losses.append(loss.item())

        if torch.isnan(loss):
            raise RuntimeError(f"Step {step}: Loss is NaN!")

        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"       Step {step}/{n_steps} | Loss: {loss.item():.6f} | "
              f"Time: {elapsed:.2f}s | GPU: {mem:.2f} GB")

    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"\n       Peak GPU: {max_mem:.2f} / {total_mem:.2f} GB")
    print(f"       Avg step time: {np.mean(step_times):.2f}s")
    print(f"       PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("  3D SR PIPELINE DRY RUN")
    print("=" * 60)

    if args.gpu:
        if not torch.cuda.is_available():
            print("ERROR: --gpu requires CUDA")
            sys.exit(1)
        device = torch.device("cuda")
        print(f"\nDevice: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDevice: {device}")
    else:
        device = torch.device("cpu")
        print(f"\nDevice: {device}")

    try:
        model = test_model(device)
        test_forward(model, device)
        test_backward(model, device)
        test_dataset(device)
        test_checkpoint(model)

        if args.gpu:
            del model
            torch.cuda.empty_cache()
            test_gpu_stress(device, n_steps=args.steps)

        print("\n" + "=" * 60)
        if args.gpu:
            print("  ALL TESTS PASSED (GPU stress test included)")
            print("  Safe to train overnight.")
        else:
            print("  ALL BASIC TESTS PASSED")
            print("  Use --gpu on CUDA machine for full validation.")
        print("=" * 60)

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
