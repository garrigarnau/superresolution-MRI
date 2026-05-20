"""Fine-tuning script for Swin2SR on MRI super-resolution data."""

import sys
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    IN_CHANNELS,
    L1_WEIGHT,
    LEARNING_RATE,
    LOG_INTERVAL,
    MODEL_NAME,
    NUM_WORKERS,
    PERCEPTUAL_WEIGHT,
    SCHEDULER_PATIENCE,
    SEED,
    TRAIN_HR_DIR,
    TRAIN_LR_DIR,
    USE_AMP,
    VAL_HR_DIR,
    VAL_INTERVAL,
    VAL_LR_DIR,
    WEIGHT_DECAY,
)
from dataset import MRISuperResDataset
from losses import CombinedLoss
from model import load_swin2sr_grayscale


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_psnr_batch(sr, hr):
    """Compute average PSNR for a batch of images."""
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    values = []
    for i in range(sr_np.shape[0]):
        sr_img = np.clip(sr_np[i, 0] * 255.0, 0, 255).astype(np.uint8)
        hr_img = (hr_np[i, 0] * 255.0).astype(np.uint8)
        values.append(psnr(hr_img, sr_img, data_range=255))
    return np.mean(values)


def forward_pass(model, lr_batch):
    """Run the Swin2SR model and extract the reconstructed image."""
    outputs = model(lr_batch)
    return outputs.reconstruction


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

        sr_batch = forward_pass(model, lr_batch)

        # Handle potential size mismatch
        if sr_batch.shape != hr_batch.shape:
            sr_batch = torch.nn.functional.interpolate(
                sr_batch, size=hr_batch.shape[2:], mode="bicubic", align_corners=False
            )

        loss, _, _ = criterion(sr_batch, hr_batch)
        total_loss += loss.item()
        total_psnr += compute_psnr_batch(sr_batch, hr_batch)
        n_batches += 1

    model.train()
    return total_loss / n_batches, total_psnr / n_batches


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Swin2SR on MRI data.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (for testing only)")
    args = parser.parse_args()

    set_seed(SEED)

    # ─── Device ───────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
        print(f"Device: {device} (forced via --cpu)")
    elif not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use --cpu for testing or run on a CUDA machine.")
        sys.exit(1)
    else:
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # ─── Data ─────────────────────────────────────────────────────────────────
    train_dataset = MRISuperResDataset(TRAIN_HR_DIR, TRAIN_LR_DIR)
    val_dataset = MRISuperResDataset(VAL_HR_DIR, VAL_LR_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Training: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"Validation: {len(val_dataset)} images ({len(val_loader)} batches)")

    # ─── Model ────────────────────────────────────────────────────────────────
    print(f"\nLoading pre-trained {MODEL_NAME}...")
    model = load_swin2sr_grayscale(MODEL_NAME)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ─── Loss, optimizer, scheduler ───────────────────────────────────────────
    criterion = CombinedLoss(l1_weight=L1_WEIGHT, perceptual_weight=PERCEPTUAL_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=SCHEDULER_PATIENCE, factor=0.5
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    # ─── Checkpointing setup ─────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_psnr = 0.0
    epochs_without_improvement = 0

    # ─── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Loss: L1 (w={L1_WEIGHT}) + Perceptual (w={PERCEPTUAL_WEIGHT})")
    print(f"LR: {LEARNING_RATE}, AMP: {USE_AMP}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_l1 = 0.0
        running_perceptual = 0.0

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader, 1):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda", enabled=USE_AMP):
                sr_batch = forward_pass(model, lr_batch)

                # Handle potential size mismatch
                if sr_batch.shape != hr_batch.shape:
                    sr_batch = torch.nn.functional.interpolate(
                        sr_batch, size=hr_batch.shape[2:], mode="bicubic", align_corners=False
                    )

                loss, l1_val, perc_val = criterion(sr_batch, hr_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_l1 += l1_val.item()
            running_perceptual += perc_val.item()

            if batch_idx % LOG_INTERVAL == 0:
                avg_loss = running_loss / batch_idx
                print(f"  Epoch {epoch}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.5f} (L1: {running_l1/batch_idx:.5f}, "
                      f"Perc: {running_perceptual/batch_idx:.5f})")

        epoch_time = time.time() - epoch_start
        avg_train_loss = running_loss / len(train_loader)

        # ─── Validation ───────────────────────────────────────────────────────
        if epoch % VAL_INTERVAL == 0:
            val_loss, val_psnr = validate(model, val_loader, criterion, device)
            scheduler.step(val_psnr)

            print(f"Epoch {epoch}/{EPOCHS} ({epoch_time:.1f}s) - "
                  f"Train Loss: {avg_train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Checkpointing
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                }, CHECKPOINT_DIR / "best_model.pth")
                print(f"  -> New best model saved (PSNR: {best_psnr:.2f} dB)")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                    break
        else:
            print(f"Epoch {epoch}/{EPOCHS} ({epoch_time:.1f}s) - "
                  f"Train Loss: {avg_train_loss:.5f}")

    # ─── Final save ───────────────────────────────────────────────────────────
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_psnr": best_psnr,
    }, CHECKPOINT_DIR / "last_model.pth")

    print("=" * 70)
    print(f"Training complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
