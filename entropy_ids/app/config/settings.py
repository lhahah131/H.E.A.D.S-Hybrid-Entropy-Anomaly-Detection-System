import os

# Base paths - Adjusted for /app/config/ location
# __file__ is /app/config/settings.py
# 1st dirname -> /app/config
# 2nd dirname -> /app
# 3rd dirname -> /entropy_ids (the project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "features") # Pointing to the new data location
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Constants
RANDOM_SEED = 42
CV_FOLDS = 5
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42

# Model settings
N_ESTIMATORS = 300
CONTAMINATION_RATIO = 0.3505

# Threshold settings
PERCENTILE_VALUE = 55
STRICT_THRESHOLD_MULTIPLIER = 1.05

# Confirmation Layer settings
ASCII_RATIO_MIN = 0.85
NON_PRINTABLE_RATIO_MAX = 0.05
GLOBAL_ENTROPY_MAX = 4.8
