"""
3D Super-Resolution model: MedicalNet ResNet3D-50 encoder + trainable 3D decoder.

Architecture:
- Encoder: Frozen MedicalNet ResNet3D-50 (extracts multi-scale 3D features)
- Decoder: Lightweight 3D upsampling path with skip connections from encoder

For input LR patch (1, 1, 32, 32, 32):
  Encoder features:
    - embedding: (1, 64, 8, 8, 8)    — after stem (conv + pool, /4)
    - stage1:    (1, 256, 8, 8, 8)    — no spatial reduction
    - stage2:    (1, 512, 4, 4, 4)    — /2
    - stage3:    (1, 1024, 2, 2, 2)   — /2
    - stage4:    (1, 2048, 1, 1, 1)   — /2

  Decoder reconstructs: (1, 1, 64, 64, 64) — x2 output
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DecoderBlock(nn.Module):
    """3D upsample block: ConvTranspose3d + Conv3d + ReLU."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Pad if sizes don't match exactly
            if x.shape != skip.shape:
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                x = nn.functional.pad(x, [0, diff_w, 0, diff_h, 0, diff_d])
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SRDecoder3D(nn.Module):
    """
    3D decoder that takes multi-scale encoder features and produces x2 upscaled output.

    Decoder path (for 32³ LR input → 64³ HR output):
        stage4 (2048, 1,1,1)  → up → (512, 2,2,2)  + stage3 skip (1024)
        → up → (256, 4,4,4)   + stage2 skip (512)
        → up → (128, 8,8,8)   + stage1 skip (256)
        → up → (64, 16,16,16) + embedding skip (64)
        → up → (32, 32,32,32)
        → up → (16, 64,64,64) — final x2 upscale
        → conv → (1, 64,64,64)
    """

    def __init__(self):
        super().__init__()
        # Decoder blocks: upsample from bottleneck back to spatial dims
        self.up4 = DecoderBlock(2048, 1024, 512)   # 1→2, concat stage3
        self.up3 = DecoderBlock(512, 512, 256)     # 2→4, concat stage2
        self.up2 = DecoderBlock(256, 256, 128)     # 4→8, concat stage1
        self.up1 = DecoderBlock(128, 64, 64)       # 8→16, concat embedding
        self.up0 = DecoderBlock(64, 0, 32)         # 16→32 (no skip)
        self.up_final = DecoderBlock(32, 0, 16)    # 32→64 (x2 upscale to HR)

        # Final 1x1 conv to single channel output
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: tuple of encoder features
                [0] embedding (64, 8, 8, 8)
                [1] stage1 (256, 8, 8, 8)
                [2] stage2 (512, 4, 4, 4)
                [3] stage3 (1024, 2, 2, 2)
                [4] stage4 (2048, 1, 1, 1)
        """
        emb, s1, s2, s3, s4 = hidden_states

        x = self.up4(s4, s3)       # (512, 2, 2, 2)
        x = self.up3(x, s2)        # (256, 4, 4, 4)
        x = self.up2(x, s1)        # (128, 8, 8, 8)
        x = self.up1(x, emb)       # (64, 16, 16, 16)
        x = self.up0(x, None)      # (32, 32, 32, 32)
        x = self.up_final(x, None) # (16, 64, 64, 64)

        return self.final_conv(x)  # (1, 64, 64, 64)


class MedicalNetSR3D(nn.Module):
    """Full 3D SR model: frozen MedicalNet encoder + trainable decoder."""

    def __init__(self, encoder_name="nwirandx/medicalnet-resnet3d50"):
        super().__init__()

        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable decoder
        self.decoder = SRDecoder3D()

    def forward(self, lr_input):
        """
        Args:
            lr_input: (B, 1, 32, 32, 32) LR volume patch

        Returns:
            sr_output: (B, 1, 64, 64, 64) SR volume patch
        """
        # Extract multi-scale features from frozen encoder
        with torch.no_grad():
            encoder_out = self.encoder(
                lr_input,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = encoder_out.hidden_states  # tuple of 5 feature maps

        # Decode to HR
        sr_output = self.decoder(hidden_states)

        return sr_output
