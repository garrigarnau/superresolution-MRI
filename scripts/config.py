"""Shared configuration for the MRI super-resolution scripts."""

from pathlib import Path

# Project directories
DATA_DIR = Path("disc1")
OUTPUT_DIR = Path("data")
RESULTS_DIR = Path("results")
TEST_LR_DIR = Path("data/test/LR")
TEST_HR_DIR = Path("data/test/HR")
WEIGHTS_DIR = Path("weights")

# Image preprocessing
HR_SIZE = 256
LR_SIZE = 64  # 4x downscale
SLICE_AXIS = 2  # 0=sagittal, 1=coronal, 2=axial
BRAIN_THRESHOLD = 0.01  # minimum fraction of non-zero pixels to keep a slice
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

# Benchmark outputs. Swin2SR is canonical and aligned to the LR footprint.
SWIN2SR_RESULTS_DIR = RESULTS_DIR / "swin2sr"
REAL_ESRGAN_RESULTS_DIR = RESULTS_DIR / "real_esrgan"
REAL_ESRGAN_WEIGHTS_PATH = WEIGHTS_DIR / "RealESRGAN_x4.pth"
MODELS = {
    "swin2sr": SWIN2SR_RESULTS_DIR,
    "real_esrgan": REAL_ESRGAN_RESULTS_DIR,
}
