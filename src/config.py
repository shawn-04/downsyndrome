"""
config.py — Single source of truth for all project-wide constants.

Every tunable value lives here so that notebooks, training scripts,
and the Streamlit app all share the same defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]       # …/down-syndrome-cnn/
DATA_DIR     = PROJECT_ROOT / "ds"
TRAIN_DIR    = DATA_DIR / "train"
VALID_DIR    = DATA_DIR / "valid"
TEST_DIR     = DATA_DIR / "test"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH  = MODEL_DIR / "best_down_syndrome_model.keras"
FINAL_MODEL_PATH = MODEL_DIR / "down_syndrome_cnn_final.keras"

# ---------------------------------------------------------------------------
# CSV label files (each split folder has a CSV with the same name)
# ---------------------------------------------------------------------------
TRAIN_CSV = TRAIN_DIR / "_classes.csv"
VALID_CSV = VALID_DIR / "_classes.csv"
TEST_CSV  = TEST_DIR  / "_classes.csv"

# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------
FILENAME_COLUMN       = "filename"
DOWN_SYNDROME_COLUMN  = "downSyndrome"     # 1 = positive
HEALTHY_COLUMN        = "healthy"          # 1 = negative

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
IMAGE_SIZE    = (IMAGE_HEIGHT, IMAGE_WIDTH)
NUM_CHANNELS  = 3
INPUT_SHAPE   = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE       = 32
NUM_EPOCHS       = 50
LEARNING_RATE    = 1e-4

# ---------------------------------------------------------------------------
# Class labels  (alphabetical order — Keras sorts labels this way)
# "Down_Syndrome" < "Healthy" alphabetically → index 0 = DS, index 1 = Healthy
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Down_Syndrome", "Healthy"]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42