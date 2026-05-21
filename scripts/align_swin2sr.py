"""Compatibility script for aligning legacy Swin2SR outputs.

The canonical Swin2SR benchmark path now applies this alignment inside
scripts/benchmark.py and writes aligned images directly to results/swin2sr/.
Use this script only to repair older unaligned result folders.
"""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image

from config import RESULTS_DIR, SWIN2SR_RESULTS_DIR, TEST_LR_DIR
from swin2sr_alignment import align_to_target_bbox


OUTPUT_SIZE = (256, 256)


def main():
    parser = argparse.ArgumentParser(
        description="Align legacy Swin2SR outputs to the LR image footprint."
    )
    parser.add_argument("--input-dir", type=Path, default=SWIN2SR_RESULTS_DIR)
    parser.add_argument("--target-lr-dir", type=Path, default=TEST_LR_DIR)
    parser.add_argument("--output-dir", type=Path, default=SWIN2SR_RESULTS_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.input_dir.glob("*.png"))

    print(f"Input: {args.input_dir}")
    print(f"Target LR footprint: {args.target_lr_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Aligning {len(image_paths)} images...")

    for index, sr_path in enumerate(image_paths, 1):
        lr_path = args.target_lr_dir / sr_path.name
        if not lr_path.exists():
            print(f"  Skipping {sr_path.name}: missing LR image")
            continue

        sr = np.array(Image.open(sr_path).convert("L"))
        lr = Image.open(lr_path).convert("L").resize(OUTPUT_SIZE, Image.NEAREST)
        target = np.array(lr)

        aligned = align_to_target_bbox(sr, target)
        Image.fromarray(aligned, mode="L").save(args.output_dir / sr_path.name)

        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"  [{index}/{len(image_paths)}]")

    print("Done.")


if __name__ == "__main__":
    main()
