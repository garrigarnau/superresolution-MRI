"""L1 + VGG Perceptual loss for super-resolution training."""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 feature maps (layers before relu1_2, relu2_2, relu3_4)."""

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Extract features at relu1_2 (idx 2), relu2_2 (idx 7), relu3_4 (idx 16)
        self.slice1 = nn.Sequential(*list(vgg.children())[:3])
        self.slice2 = nn.Sequential(*list(vgg.children())[3:8])
        self.slice3 = nn.Sequential(*list(vgg.children())[8:17])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        # Replicate 1-channel grayscale to 3 channels for VGG
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)

        sr_f1 = self.slice1(sr)
        sr_f2 = self.slice2(sr_f1)
        sr_f3 = self.slice3(sr_f2)

        hr_f1 = self.slice1(hr)
        hr_f2 = self.slice2(hr_f1)
        hr_f3 = self.slice3(hr_f2)

        loss = (
            nn.functional.l1_loss(sr_f1, hr_f1)
            + nn.functional.l1_loss(sr_f2, hr_f2)
            + nn.functional.l1_loss(sr_f3, hr_f3)
        )
        return loss


class CombinedLoss(nn.Module):
    """L1 + weighted perceptual loss."""

    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, sr, hr):
        l1 = self.l1_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)
        return self.l1_weight * l1 + self.perceptual_weight * perceptual, l1, perceptual
