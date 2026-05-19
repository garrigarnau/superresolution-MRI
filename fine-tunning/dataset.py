"""PyTorch Dataset for MRI HR/LR image pairs."""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np


class MRISuperResDataset(Dataset):
    """Loads paired HR/LR PNG images for super-resolution training."""

    def __init__(self, hr_dir: Path, lr_dir: Path):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.filenames = sorted([f.name for f in self.hr_dir.glob("*.png")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        hr_img = np.array(Image.open(self.hr_dir / fname).convert("L"), dtype=np.float32)
        lr_img = np.array(Image.open(self.lr_dir / fname).convert("L"), dtype=np.float32)

        # Normalize to [0, 1] and add channel dim: (1, H, W)
        hr_tensor = torch.from_numpy(hr_img / 255.0).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_img / 255.0).unsqueeze(0)

        return lr_tensor, hr_tensor
