"""Align Swin2SR benchmark outputs to the LR image footprint.

The HuggingFace RGB Swin2SR baseline can shift/scale the foreground for these
MRI slices. This postprocess keeps the model output but maps its foreground
bounding box back onto the LR image footprint, so PSNR/SSIM compare aligned
images instead of penalizing a systematic spatial mismatch.
"""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image

from config import RESULTS_DIR, SWIN2SR_RESULTS_DIR, TEST_LR_DIR


OUTPUT_DIR = RESULTS_DIR / "swin2sr_aligned"
THRESHOLD = 5
OUTPUT_SIZE = (256, 256)


def foreground_bbox(image, threshold=THRESHOLD):
    ys, xs = np.where(image > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def align_to_target_bbox(source, target):
    source_bbox = foreground_bbox(source)
    target_bbox = foreground_bbox(target)
    if source_bbox is None or target_bbox is None:
        return source

    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox

    source_crop = Image.fromarray(source[sy0:sy1, sx0:sx1], mode="L")
    source_crop = source_crop.resize((tx1 - tx0, ty1 - ty0), Image.BICUBIC)

    aligned = Image.new("L", OUTPUT_SIZE, 0)
    aligned.paste(source_crop, (tx0, ty0))
    return np.array(aligned)


def main():
    parser = argparse.ArgumentParser(description="Align Swin2SR outputs to the LR image footprint.")
    parser.add_argument("--input-dir", type=Path, default=SWIN2SR_RESULTS_DIR)
    parser.add_argument("--target-lr-dir", type=Path, default=TEST_LR_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
