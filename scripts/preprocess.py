"""
Preprocessing script for MRI Super-Resolution project.
Converts 3D Analyze (.hdr/.img) volumes to 2D PNG slices,
generates HR/LR pairs, and splits into train/val/test.
"""

import numpy as np
import nibabel as nib
from PIL import Image
import random

from config import (
    BRAIN_THRESHOLD,
    DATA_DIR,
    HR_SIZE,
    LR_SIZE,
    OUTPUT_DIR,
    SEED,
    SLICE_AXIS,
    TRAIN_RATIO,
    VAL_RATIO,
)


def load_volume(hdr_path):
    """Load an Analyze format volume."""
    img = nib.load(str(hdr_path))
    data = img.get_fdata()
    return np.squeeze(data)


def normalize_slice(slice_2d):
    """Normalize a 2D slice to 0-255 uint8."""
    s = slice_2d.astype(np.float64)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min == 0:
        return np.zeros_like(s, dtype=np.uint8)
    s = (s - s_min) / (s_max - s_min) * 255.0
    return s.astype(np.uint8)


def is_valid_slice(slice_2d, threshold=BRAIN_THRESHOLD):
    """Check if slice has enough brain content."""
    non_zero = np.count_nonzero(slice_2d)
    total = slice_2d.size
    return (non_zero / total) > threshold


def extract_slices(volume):
    """Extract valid axial slices from volume."""
    slices = []
    n_slices = volume.shape[SLICE_AXIS]

    for i in range(n_slices):
        if SLICE_AXIS == 0:
            s = volume[i, :, :]
        elif SLICE_AXIS == 1:
            s = volume[:, i, :]
        else:
            s = volume[:, :, i]

        if is_valid_slice(s):
            slices.append(normalize_slice(s))

    return slices


def create_hr_lr_pair(slice_2d):
    """Create HR (256x256) and LR (64x64) pair from a slice."""
    img = Image.fromarray(slice_2d, mode='L')
    hr = img.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
    lr = hr.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
    return np.array(hr), np.array(lr)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Find all masked_gfc volumes (brain-extracted, bias-corrected, Talairach space)
    subject_dirs = sorted(DATA_DIR.glob("OAS1_*_MR1"))
    print(f"Found {len(subject_dirs)} subjects")

    # Collect all slices
    all_slices = []  # list of (subject_id, slice_idx, hr, lr)

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        # Use the masked, bias-corrected, Talairach-transformed volume
        hdr_files = list(subj_dir.glob("PROCESSED/MPRAGE/T88_111/*_masked_gfc.hdr"))

        if not hdr_files:
            print(f"  Skipping {subj_id}: no masked_gfc volume found")
            continue

        hdr_path = hdr_files[0]
        print(f"  Processing {subj_id}: {hdr_path.name}")

        volume = load_volume(hdr_path)
        slices = extract_slices(volume)
        print(f"    Extracted {len(slices)} valid slices (shape: {volume.shape})")

        for idx, s in enumerate(slices):
            hr, lr = create_hr_lr_pair(s)
            all_slices.append((subj_id, idx, hr, lr))

    print(f"\nTotal slices collected: {len(all_slices)}")

    # Shuffle and split
    random.shuffle(all_slices)
    n_total = len(all_slices)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_slices = all_slices[:n_train]
    val_slices = all_slices[n_train:n_train + n_val]
    test_slices = all_slices[n_train + n_val:]

    print(f"Split: train={len(train_slices)}, val={len(val_slices)}, test={len(test_slices)}")

    # Save to disk
    for split_name, split_data in [("train", train_slices), ("val", val_slices), ("test", test_slices)]:
        hr_dir = OUTPUT_DIR / split_name / "HR"
        lr_dir = OUTPUT_DIR / split_name / "LR"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

        for i, (subj_id, slice_idx, hr, lr) in enumerate(split_data):
            filename = f"{subj_id}_slice{slice_idx:03d}.png"
            Image.fromarray(hr, mode='L').save(hr_dir / filename)
            Image.fromarray(lr, mode='L').save(lr_dir / filename)

        print(f"  Saved {len(split_data)} images to {OUTPUT_DIR / split_name}/")

    print("\nDone! Output structure:")
    print("  data/train/HR/  - High-resolution 256x256 images")
    print("  data/train/LR/  - Low-resolution 64x64 images")
    print("  data/val/HR/    data/val/LR/")
    print("  data/test/HR/   data/test/LR/")


if __name__ == "__main__":
    main()
