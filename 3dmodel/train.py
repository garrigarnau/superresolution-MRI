"""Training script for 3D MRI Super-Resolution model."""

import sys
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    ENCODER_NAME,
    EPOCHS,
    LEARNING_RATE,
    LOG_INTERVAL,
    NUM_WORKERS,
    PATCHES_DIR,
    SCHEDULER_PATIENCE,
    SEED,
    WEIGHT_DECAY,
)
from dataset import MRI3DPatchDataset
from model import MedicalNetSR3D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr_batch(sr, hr):
    """Compute average PSNR for a batch of 3D patches."""
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    values = []
    for i in range(sr_np.shape[0]):
        sr_vol = np.clip(sr_np[i, 0], 0, 1)
        hr_vol = hr_np[i, 0]
        values.append(psnr(hr_vol, sr_vol, data_range=1.0))
    return np.mean(values)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run validation and return average loss and PSNR."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n_batches = 0

    for lr_batch, hr_batch in val_loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        sr_batch = model(lr_batch)
        loss = criterion(sr_batch, hr_batch)

        total_loss += loss.item()
        total_psnr += compute_psnr_batch(sr_batch, hr_batch)
        n_batches += 1

    model.train()
    return total_loss / n_batches, total_psnr / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train 3D SR model.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (testing only)")
    args = parser.parse_args()

    set_seed(SEED)

    # ─── Device ───────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use --cpu for testing.")
        sys.exit(1)
    else:
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # ─── Data ─────────────────────────────────────────────────────────────────
    train_dataset = MRI3DPatchDataset(PATCHES_DIR / "train")
    val_dataset = MRI3DPatchDataset(PATCHES_DIR / "val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Training: {len(train_dataset)} patches ({len(train_loader)} batches)")
    print(f"Validation: {len(val_dataset)} patches ({len(val_loader)} batches)")

    # ─── Model ────────────────────────────────────────────────────────────────
    print(f"\nLoading MedicalNet encoder: {ENCODER_NAME}...")
    model = MedicalNetSR3D(ENCODER_NAME).to(device)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder: {encoder_params:,} params (frozen)")
    print(f"Decoder: {decoder_params:,} params (trainable)")
    print(f"Total trainable: {trainable_params:,}")

    # ─── Loss, optimizer, scheduler ───────────────────────────────────────────
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=SCHEDULER_PATIENCE, factor=0.5
    )

    # ─── Checkpointing ───────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_psnr = 0.0
    epochs_without_improvement = 0

    # ─── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Loss: L1 | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        running_loss = 0.0
        model.train()

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader, 1):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % LOG_INTERVAL == 0:
                avg_loss = running_loss / batch_idx
                print(f"  Epoch {epoch}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.6f}")

        epoch_time = time.time() - epoch_start
        avg_train_loss = running_loss / len(train_loader)

        # ─── Validation ───────────────────────────────────────────────────────
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        scheduler.step(val_psnr)

        print(f"Epoch {epoch}/{EPOCHS} ({epoch_time:.1f}s) — "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Checkpointing
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
            }, CHECKPOINT_DIR / "best_model.pth")
            print(f"  -> New best model saved (PSNR: {best_psnr:.2f} dB)")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    # ─── Final save ───────────────────────────────────────────────────────────
    torch.save({
        "epoch": epoch,
        "decoder_state_dict": model.decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_psnr": best_psnr,
    }, CHECKPOINT_DIR / "last_model.pth")

    print("=" * 60)
    print(f"Training complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
