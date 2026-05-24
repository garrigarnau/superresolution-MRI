"""PyTorch Dataset for 3D HR/LR patch pairs."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class MRI3DPatchDataset(Dataset):
    """Loads paired 3D HR/LR patches stored as .npy files."""

    def __init__(self, patches_dir: Path):
        self.patches_dir = Path(patches_dir)
        self.hr_dir = self.patches_dir / "HR"
        self.lr_dir = self.patches_dir / "LR"
        self.filenames = sorted([f.name for f in self.hr_dir.glob("*.npy")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        hr_patch = np.load(self.hr_dir / fname).astype(np.float32)
        lr_patch = np.load(self.lr_dir / fname).astype(np.float32)

        # Add channel dimension: (1, D, H, W)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)

        return lr_tensor, hr_tensor
