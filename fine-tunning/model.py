"""Load and adapt Swin2SR for 1-channel (grayscale) MRI super-resolution."""

import torch
import torch.nn as nn
from transformers import Swin2SRForImageSuperResolution, Swin2SRConfig


def load_swin2sr_grayscale(model_name: str = "caidas/swin2SR-classical-sr-x4-64"):
    """
    Load pre-trained Swin2SR and modify input/output layers for 1-channel images.

    Strategy:
    - First conv: Conv2d(3, 180) → Conv2d(1, 180). Initialize by averaging the 3-channel weights.
    - Final conv: Conv2d(64, 3) → Conv2d(64, 1). Initialize by averaging the 3-channel weights.
    - Mean tensor: (1, 3, 1, 1) → (1, 1, 1, 1). Average the RGB means.
    """
    model = Swin2SRForImageSuperResolution.from_pretrained(model_name)

    # ─── Modify first convolution: 3ch → 1ch input ───────────────────────────
    old_first_conv = model.swin2sr.first_convolution
    new_first_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_first_conv.out_channels,
        kernel_size=old_first_conv.kernel_size,
        stride=old_first_conv.stride,
        padding=old_first_conv.padding,
    )
    # Average the 3 input channel weights → single channel
    with torch.no_grad():
        new_first_conv.weight.copy_(old_first_conv.weight.mean(dim=1, keepdim=True))
        new_first_conv.bias.copy_(old_first_conv.bias)
    model.swin2sr.first_convolution = new_first_conv

    # ─── Modify final convolution: 3ch → 1ch output ──────────────────────────
    old_final_conv = model.upsample.final_convolution
    new_final_conv = nn.Conv2d(
        in_channels=old_final_conv.in_channels,
        out_channels=1,
        kernel_size=old_final_conv.kernel_size,
        stride=old_final_conv.stride,
        padding=old_final_conv.padding,
    )
    # Average the 3 output channel weights → single channel
    with torch.no_grad():
        new_final_conv.weight.copy_(old_final_conv.weight.mean(dim=0, keepdim=True))
        new_final_conv.bias.copy_(old_final_conv.bias.mean(dim=0, keepdim=True))
    model.upsample.final_convolution = new_final_conv

    # ─── Update the mean tensor used for input normalization ──────────────────
    # Original: shape (1, 3, 1, 1) with RGB means
    old_mean = model.swin2sr.mean  # registered buffer
    new_mean = old_mean.mean(dim=1, keepdim=True)  # (1, 1, 1, 1)
    model.swin2sr.mean = nn.Parameter(new_mean, requires_grad=False)

    # ─── Update config to reflect 1 channel ───────────────────────────────────
    model.config.num_channels = 1
    model.config.num_channels_out = 1

    return model
