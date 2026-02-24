import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "datasets")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Constants
RANDOM_SEED = 42
CV_FOLDS = 5

# Model settings
N_ESTIMATORS = 300
CONTAMINATION_RATIO = 0.18

# Threshold settings
PERCENTILE_VALUE = 56
STRICT_THRESHOLD_MULTIPLIER = 1.05

# Confirmation Layer settings
ASCII_RATIO_MIN = 0.85
NON_PRINTABLE_RATIO_MAX = 0.05
GLOBAL_ENTROPY_MAX = 4.8
