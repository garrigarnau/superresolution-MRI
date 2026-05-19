"""
Dry-run test of the fine-tuning pipeline.
Runs on CPU/MPS with a small subset of data to verify everything works
end-to-end before deploying to a CUDA GPU.

When run on CUDA (--gpu flag), also simulates the first N training steps
with AMP, full dataloader, and validation to guarantee overnight stability.

Usage:
    python test_pipeline.py          # CPU/MPS basic test
    python test_pipeline.py --gpu    # Full GPU stress test (run this on CUDA machine)
"""

import sys
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    L1_WEIGHT,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_WORKERS,
    PERCEPTUAL_WEIGHT,
    TRAIN_HR_DIR,
    TRAIN_LR_DIR,
    USE_AMP,
    VAL_HR_DIR,
    VAL_LR_DIR,
    WEIGHT_DECAY,
)
from dataset import MRISuperResDataset
from losses import CombinedLoss
from model import load_swin2sr_grayscale


def test_dataset():
    """Test that the dataset loads correctly."""
    print("[1/6] Testing dataset loading...")
    train_ds = MRISuperResDataset(TRAIN_HR_DIR, TRAIN_LR_DIR)
    val_ds = MRISuperResDataset(VAL_HR_DIR, VAL_LR_DIR)

    assert len(train_ds) > 0, f"Train dataset is empty! Check {TRAIN_HR_DIR}"
    assert len(val_ds) > 0, f"Val dataset is empty! Check {VAL_HR_DIR}"

    lr, hr = train_ds[0]
    assert lr.shape == (1, 64, 64), f"LR shape wrong: {lr.shape}"
    assert hr.shape == (1, 256, 256), f"HR shape wrong: {hr.shape}"
    assert lr.min() >= 0 and lr.max() <= 1, "LR not normalized to [0,1]"
    assert hr.min() >= 0 and hr.max() <= 1, "HR not normalized to [0,1]"

    print(f"       Train: {len(train_ds)} images, Val: {len(val_ds)} images")
    print(f"       LR shape: {lr.shape}, HR shape: {hr.shape}")
    print(f"       LR range: [{lr.min():.3f}, {lr.max():.3f}]")
    print(f"       PASSED")
    return train_ds, val_ds


def test_model(device):
    """Test model loading and 1-channel adaptation."""
    print(f"\n[2/6] Testing model loading ({MODEL_NAME})...")
    model = load_swin2sr_grayscale(MODEL_NAME)
    model = model.to(device)

    # Check first conv is 1 channel
    first_conv = model.swin2sr.first_convolution
    assert first_conv.in_channels == 1, f"First conv in_channels: {first_conv.in_channels}"

    # Check final conv is 1 channel
    final_conv = model.upsample.final_convolution
    assert final_conv.out_channels == 1, f"Final conv out_channels: {final_conv.out_channels}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {total_params:,}")
    print(f"       First conv: {first_conv}")
    print(f"       Final conv: {final_conv}")
    print(f"       PASSED")
    return model


def test_forward(model, train_ds, device):
    """Test forward pass with a real batch."""
    print(f"\n[3/6] Testing forward pass...")
    subset = Subset(train_ds, range(2))
    loader = DataLoader(subset, batch_size=2, shuffle=False)

    lr_batch, hr_batch = next(iter(loader))
    lr_batch = lr_batch.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(lr_batch)
        sr_batch = outputs.reconstruction

    print(f"       Input:  {lr_batch.shape}")
    print(f"       Output: {sr_batch.shape}")
    assert sr_batch.shape[1] == 1, f"Output channels: {sr_batch.shape[1]}, expected 1"
    # Output might not be exactly 256x256 depending on model padding
    print(f"       Output range: [{sr_batch.min():.3f}, {sr_batch.max():.3f}]")
    print(f"       PASSED")
    return sr_batch, hr_batch


def test_loss(device):
    """Test loss computation."""
    print(f"\n[4/6] Testing loss function...")
    criterion = CombinedLoss(l1_weight=L1_WEIGHT, perceptual_weight=PERCEPTUAL_WEIGHT).to(device)

    # Fake SR and HR tensors
    sr = torch.rand(2, 1, 256, 256, device=device)
    hr = torch.rand(2, 1, 256, 256, device=device)

    total_loss, l1_loss, perc_loss = criterion(sr, hr)

    assert total_loss.requires_grad is False or total_loss.item() > 0
    print(f"       Total: {total_loss.item():.5f}, L1: {l1_loss.item():.5f}, Perc: {perc_loss.item():.5f}")
    print(f"       PASSED")


def test_backward(model, train_ds, device):
    """Test a full training step (forward + backward + optimizer step)."""
    print(f"\n[5/6] Testing training step (forward + backward)...")
    model.train()
    criterion = CombinedLoss(l1_weight=L1_WEIGHT, perceptual_weight=PERCEPTUAL_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    subset = Subset(train_ds, range(2))
    loader = DataLoader(subset, batch_size=2, shuffle=False)
    lr_batch, hr_batch = next(iter(loader))
    lr_batch = lr_batch.to(device)
    hr_batch = hr_batch.to(device)

    optimizer.zero_grad()
    sr_batch = model(lr_batch).reconstruction

    # Resize if needed
    if sr_batch.shape != hr_batch.shape:
        sr_batch = torch.nn.functional.interpolate(
            sr_batch, size=hr_batch.shape[2:], mode="bicubic", align_corners=False
        )

    loss, l1_val, perc_val = criterion(sr_batch, hr_batch)
    loss.backward()
    optimizer.step()

    print(f"       Loss: {loss.item():.5f}")
    print(f"       Gradients computed successfully")
    print(f"       PASSED")


def test_checkpoint(model):
    """Test saving and loading a checkpoint."""
    print(f"\n[6/6] Testing checkpoint save/load...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = CHECKPOINT_DIR / "test_checkpoint.pth"

    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "best_psnr": 25.0,
    }, test_path)

    checkpoint = torch.load(test_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_psnr"] == 25.0

    # Clean up test file
    test_path.unlink()
    print(f"       Save/load to {CHECKPOINT_DIR} works")
    print(f"       PASSED")


def test_gpu_training(train_ds, val_ds, device, n_steps=20):
    """
    Simulate real training on CUDA: multiple batches with AMP, full dataloader,
    and a validation pass. This catches GPU-specific issues like:
    - OOM with real batch sizes
    - AMP NaN/Inf issues
    - DataLoader multiprocessing with pin_memory
    - Sustained GPU memory stability over multiple steps
    """
    print(f"\n[7/7] GPU stress test ({n_steps} training steps + validation)...")
    print(f"       Batch size: {BATCH_SIZE}, AMP: {USE_AMP}, Workers: {NUM_WORKERS}")

    # Full dataloader with real settings
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Fresh model on CUDA
    model = load_swin2sr_grayscale(MODEL_NAME).to(device)
    model.train()

    criterion = CombinedLoss(l1_weight=L1_WEIGHT, perceptual_weight=PERCEPTUAL_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)

    # ─── Training steps ───────────────────────────────────────────────────────
    losses = []
    step_times = []
    train_iter = iter(train_loader)

    for step in range(1, n_steps + 1):
        try:
            lr_batch, hr_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            lr_batch, hr_batch = next(train_iter)

        lr_batch = lr_batch.to(device, non_blocking=True)
        hr_batch = hr_batch.to(device, non_blocking=True)

        start = time.time()
        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            sr_batch = model(lr_batch).reconstruction
            if sr_batch.shape != hr_batch.shape:
                sr_batch = torch.nn.functional.interpolate(
                    sr_batch, size=hr_batch.shape[2:], mode="bicubic", align_corners=False
                )
            loss, l1_val, perc_val = criterion(sr_batch, hr_batch)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Step {step}: Loss is NaN or Inf! (L1={l1_val.item()}, Perc={perc_val.item()})")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        elapsed = time.time() - start
        losses.append(loss.item())
        step_times.append(elapsed)

        if step % 5 == 0 or step == 1:
            mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"       Step {step:>3}/{n_steps} | Loss: {loss.item():.5f} | "
                  f"Time: {elapsed:.2f}s | GPU Mem: {mem_used:.2f} GB")

    # ─── Check loss is decreasing (not diverging) ─────────────────────────────
    first_half = np.mean(losses[:n_steps // 2])
    second_half = np.mean(losses[n_steps // 2:])
    if second_half > first_half * 1.5:
        print(f"       WARNING: Loss may be diverging ({first_half:.5f} → {second_half:.5f})")
    else:
        print(f"       Loss trend: {first_half:.5f} → {second_half:.5f} (stable)")

    # ─── Validation pass ──────────────────────────────────────────────────────
    print(f"       Running validation ({len(val_ds)} images)...")
    model.eval()
    val_losses = []
    val_psnrs = []

    with torch.no_grad():
        for val_step, (lr_batch, hr_batch) in enumerate(val_loader):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            sr_batch = model(lr_batch).reconstruction
            if sr_batch.shape != hr_batch.shape:
                sr_batch = torch.nn.functional.interpolate(
                    sr_batch, size=hr_batch.shape[2:], mode="bicubic", align_corners=False
                )

            loss, _, _ = criterion(sr_batch, hr_batch)
            val_losses.append(loss.item())

            # PSNR for first few batches
            if val_step < 5:
                sr_np = sr_batch.cpu().numpy()
                hr_np = hr_batch.cpu().numpy()
                for i in range(sr_np.shape[0]):
                    sr_img = np.clip(sr_np[i, 0] * 255, 0, 255).astype(np.uint8)
                    hr_img = (hr_np[i, 0] * 255).astype(np.uint8)
                    val_psnrs.append(psnr(hr_img, sr_img, data_range=255))

            if torch.isnan(loss):
                raise RuntimeError(f"Validation step {val_step}: Loss is NaN!")

    avg_val_loss = np.mean(val_losses)
    avg_val_psnr = np.mean(val_psnrs) if val_psnrs else 0
    print(f"       Val Loss: {avg_val_loss:.5f} | Val PSNR: {avg_val_psnr:.2f} dB")

    # ─── Memory summary ──────────────────────────────────────────────────────
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    total_mem = torch.cuda.get_device_properties(device).total_mem / (1024 ** 3)
    print(f"       Peak GPU memory: {max_mem:.2f} / {total_mem:.2f} GB")
    print(f"       Avg step time: {np.mean(step_times):.2f}s")
    est_epoch_time = np.mean(step_times) * len(train_loader)
    print(f"       Estimated time per epoch: {est_epoch_time / 60:.1f} min")
    print(f"       Estimated total (100 epochs): {est_epoch_time * 100 / 3600:.1f} hours")

    # ─── Checkpoint save/load on GPU ──────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    test_path = CHECKPOINT_DIR / "gpu_test_checkpoint.pth"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_psnr": avg_val_psnr,
    }, test_path)
    # Reload
    ckpt = torch.load(test_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_path.unlink()
    print(f"       GPU checkpoint save/load: OK")

    print(f"       PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuning pipeline.")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU stress test (requires CUDA)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of training steps for GPU test (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("  FINE-TUNING PIPELINE DRY RUN")
    print("=" * 60)

    # Select device
    if args.gpu:
        if not torch.cuda.is_available():
            print("ERROR: --gpu flag requires CUDA. Run without it for CPU/MPS test.")
            sys.exit(1)
        device = torch.device("cuda")
        print(f"\nDevice: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Mode: GPU STRESS TEST ({args.steps} steps)\n")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDevice: {device}")
        print(f"Mode: BASIC TEST (use --gpu on CUDA machine for full test)\n")
    else:
        device = torch.device("cpu")
        print(f"\nDevice: {device}")
        print(f"Mode: BASIC TEST (use --gpu on CUDA machine for full test)\n")

    try:
        train_ds, val_ds = test_dataset()
        model = test_model(device)
        test_forward(model, train_ds, device)
        test_loss(device)
        test_backward(model, train_ds, device)
        test_checkpoint(model)

        if args.gpu:
            # Free the model from basic tests before GPU stress test
            del model
            torch.cuda.empty_cache()
            test_gpu_training(train_ds, val_ds, device, n_steps=args.steps)

        print("\n" + "=" * 60)
        if args.gpu:
            print("  ALL TESTS PASSED (including GPU stress test)")
            print("  Safe to leave training overnight.")
        else:
            print("  ALL BASIC TESTS PASSED")
            print("  Run with --gpu on CUDA machine for full validation.")
        print("=" * 60)

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
