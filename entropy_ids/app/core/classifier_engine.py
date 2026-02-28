import numpy as np
import pandas as pd
import logging
from config.settings import ASCII_RATIO_MIN, NON_PRINTABLE_RATIO_MAX, GLOBAL_ENTROPY_MAX

logger = logging.getLogger(__name__)

def classify_predictions(scores: np.ndarray, threshold: float, mode: str, scaler_roc=None) -> np.ndarray:
    """
    Classify based on threshold mode.
    If 'adaptive': scores < threshold means anomaly (1)
    If 'strict': normalized inverted scores >= threshold means anomaly (1)
    """
    logger.info(f"Classifying predictions using {mode.upper()} mode thresholding.")
    if mode == "adaptive":
        preds = (scores < threshold).astype(int)
    elif mode == "strict":
        if scaler_roc is None:
            raise ValueError("Strict mode requires a fitted scaler.")
        anomaly_scores = 1 - scaler_roc.transform(scores.reshape(-1, 1)).flatten()
        preds = (anomaly_scores >= threshold).astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return preds

from core.hwcl_engine import apply_hwcl_confirmation
