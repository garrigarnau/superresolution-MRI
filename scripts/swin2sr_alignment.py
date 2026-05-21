"""Alignment helpers for the canonical Swin2SR MRI benchmark output."""

import numpy as np
from PIL import Image


THRESHOLD = 5
OUTPUT_SIZE = (256, 256)


def foreground_bbox(image, threshold=THRESHOLD):
    ys, xs = np.where(image > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def align_to_target_bbox(source, target, output_size=OUTPUT_SIZE):
    """Map the source foreground box onto the target foreground box."""
    source_bbox = foreground_bbox(source)
    target_bbox = foreground_bbox(target)
    if source_bbox is None or target_bbox is None:
        return source

    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox

    source_crop = Image.fromarray(source[sy0:sy1, sx0:sx1], mode="L")
    source_crop = source_crop.resize((tx1 - tx0, ty1 - ty0), Image.BICUBIC)

    aligned = Image.new("L", output_size, 0)
    aligned.paste(source_crop, (tx0, ty0))
    return np.array(aligned)


def align_to_lr_footprint(source, lr_image, output_size=OUTPUT_SIZE):
    """Align an SR output to the upscaled LR foreground footprint."""
    target = lr_image.convert("L").resize(output_size, Image.NEAREST)
    return align_to_target_bbox(source, np.array(target), output_size=output_size)
