"""Configuration for Swin2SR fine-tuning on MRI super-resolution."""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_HR_DIR = PROJECT_ROOT / "data" / "train" / "HR"
TRAIN_LR_DIR = PROJECT_ROOT / "data" / "train" / "LR"
VAL_HR_DIR = PROJECT_ROOT / "data" / "val" / "HR"
VAL_LR_DIR = PROJECT_ROOT / "data" / "val" / "LR"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "caidas/swin2SR-classical-sr-x4-64"
SCALE_FACTOR = 4
IN_CHANNELS = 1  # grayscale MRI

# ─── Training hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 15

# ─── Loss weights ────────────────────────────────────────────────────────────
L1_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.1

# ─── Mixed precision ─────────────────────────────────────────────────────────
USE_AMP = True

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_INTERVAL = 50  # print every N batches
VAL_INTERVAL = 1   # validate every N epochs
SEED = 42
