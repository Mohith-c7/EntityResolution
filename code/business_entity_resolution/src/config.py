"""
Project-level configuration and constants for Amazon Business Entity Resolution 2026.
"""

from pathlib import Path

# Project root directory: 3 levels up from src/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Dataset directories
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_PATH = DATASET_DIR / "train"
TEST_PATH = DATASET_DIR / "test"

# Output and artifact directories
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Reproducibility & Model Constants
RANDOM_SEED: int = 42
TOP_K: int = 20
