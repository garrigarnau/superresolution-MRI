"""
Preprocessing for 3D MRI Super-Resolution.
Loads OASIS volumes, generates LR (x2 gaussian blur + downsample),
extracts 3D patch pairs, and splits by subject into train/val/test.
"""

import random
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom

from config import (
    BRAIN_THRESHOLD,
    DATA_DIR,
    HR_PATCH_SIZE,
    LR_PATCH_SIZE,
    PATCHES_DIR,
    PATCH_STRIDE,
    SCALE_FACTOR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def load_volume(hdr_path):
    """Load an Analyze format volume and normalize to [0, 1]."""
    img = nib.load(str(hdr_path))
    data = np.squeeze(img.get_fdata()).astype(np.float32)
    # Normalize to [0, 1]
    d_min, d_max = data.min(), data.max()
    if d_max - d_min > 0:
        data = (data - d_min) / (d_max - d_min)
    return data


def generate_lr_volume(hr_volume, scale_factor=SCALE_FACTOR):
    """Generate LR volume: Gaussian blur + downsample by scale_factor."""
    # Gaussian blur with sigma proportional to scale
    sigma = 0.5 * scale_factor
    blurred = gaussian_filter(hr_volume, sigma=sigma)
    # Downsample with trilinear interpolation
    lr_volume = zoom(blurred, 1.0 / scale_factor, order=1)
    return lr_volume


def extract_patches(hr_volume, lr_volume, patch_size_hr, patch_size_lr, stride):
    """Extract paired 3D patches from HR and LR volumes."""
    patches = []
    d_hr, h_hr, w_hr = hr_volume.shape
    d_lr, h_lr, w_lr = lr_volume.shape

    # Iterate over LR volume coordinates
    for d in range(0, d_lr - patch_size_lr + 1, stride // SCALE_FACTOR):
        for h in range(0, h_lr - patch_size_lr + 1, stride // SCALE_FACTOR):
            for w in range(0, w_lr - patch_size_lr + 1, stride // SCALE_FACTOR):
                lr_patch = lr_volume[
                    d:d + patch_size_lr,
                    h:h + patch_size_lr,
                    w:w + patch_size_lr,
                ]

                # Corresponding HR patch
                d_hr_start = d * SCALE_FACTOR
                h_hr_start = h * SCALE_FACTOR
                w_hr_start = w * SCALE_FACTOR
                hr_patch = hr_volume[
                    d_hr_start:d_hr_start + patch_size_hr,
                    h_hr_start:h_hr_start + patch_size_hr,
                    w_hr_start:w_hr_start + patch_size_hr,
                ]

                # Verify shapes
                if hr_patch.shape != (patch_size_hr,) * 3:
                    continue
                if lr_patch.shape != (patch_size_lr,) * 3:
                    continue

                # Skip patches with too little brain content
                if np.count_nonzero(hr_patch) / hr_patch.size < BRAIN_THRESHOLD:
                    continue

                patches.append((hr_patch, lr_patch))

    return patches


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Find all subjects
    subject_dirs = sorted(DATA_DIR.glob("OAS1_*_MR1"))
    print(f"Found {len(subject_dirs)} subjects")

    # Split subjects into train/val/test
    random.shuffle(subject_dirs)
    n_total = len(subject_dirs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": subject_dirs[:n_train],
        "val": subject_dirs[n_train:n_train + n_val],
        "test": subject_dirs[n_train + n_val:],
    }

    print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Process each split
    for split_name, subjects in splits.items():
        hr_dir = PATCHES_DIR / split_name / "HR"
        lr_dir = PATCHES_DIR / split_name / "LR"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

        patch_count = 0

        for subj_dir in subjects:
            subj_id = subj_dir.name
            hdr_files = list(subj_dir.glob("PROCESSED/MPRAGE/T88_111/*_masked_gfc.hdr"))

            if not hdr_files:
                print(f"  Skipping {subj_id}: no masked_gfc volume")
                continue

            print(f"  Processing {subj_id}...")
            hr_volume = load_volume(hdr_files[0])
            lr_volume = generate_lr_volume(hr_volume)

            patches = extract_patches(
                hr_volume, lr_volume,
                HR_PATCH_SIZE, LR_PATCH_SIZE, PATCH_STRIDE
            )

            for hr_patch, lr_patch in patches:
                fname = f"{subj_id}_patch{patch_count:05d}.npy"
                np.save(hr_dir / fname, hr_patch)
                np.save(lr_dir / fname, lr_patch)
                patch_count += 1

            print(f"    {len(patches)} patches extracted (total: {patch_count})")

        print(f"  {split_name}: {patch_count} patches saved\n")

    # Also save full test volumes for inference evaluation
    test_volumes_dir = PATCHES_DIR / "test_volumes"
    test_volumes_dir.mkdir(parents=True, exist_ok=True)

    for subj_dir in splits["test"]:
        subj_id = subj_dir.name
        hdr_files = list(subj_dir.glob("PROCESSED/MPRAGE/T88_111/*_masked_gfc.hdr"))
        if not hdr_files:
            continue

        hr_volume = load_volume(hdr_files[0])
        lr_volume = generate_lr_volume(hr_volume)

        np.save(test_volumes_dir / f"{subj_id}_hr.npy", hr_volume)
        np.save(test_volumes_dir / f"{subj_id}_lr.npy", lr_volume)
        print(f"  Saved test volume: {subj_id}")

    print("\nDone! Patches saved to:", PATCHES_DIR)


if __name__ == "__main__":
    main()
