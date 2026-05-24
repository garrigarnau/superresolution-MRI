"""Configuration for 3D MRI Super-Resolution with MedicalNet ResNet3D-50."""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "disc1"
PATCHES_DIR = Path(__file__).resolve().parent / "patches"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ─── Model ───────────────────────────────────────────────────────────────────
ENCODER_NAME = "nwirandx/medicalnet-resnet3d50"
SCALE_FACTOR = 2

# ─── Volume dimensions (OASIS T88_111 volumes) ──────────────────────────────
VOLUME_SHAPE = (176, 208, 176)  # original HR volume shape (D, H, W)
LR_SHAPE = (88, 104, 88)        # x2 downsampled

# ─── Patch extraction ────────────────────────────────────────────────────────
HR_PATCH_SIZE = 64   # 64x64x64 HR patches
LR_PATCH_SIZE = 32   # 32x32x32 LR patches (x2)
PATCH_STRIDE = 48    # stride for patch extraction (overlap = 64 - 48 = 16)
BRAIN_THRESHOLD = 0.01  # minimum fraction of non-zero voxels in a patch

# ─── Data split ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15
SEED = 42

# ─── Training hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 12

# ─── Inference (sliding window) ──────────────────────────────────────────────
INFER_PATCH_SIZE = 32       # LR patch size for sliding window
INFER_PATCH_OVERLAP = 8     # overlap between patches during inference

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_INTERVAL = 20
