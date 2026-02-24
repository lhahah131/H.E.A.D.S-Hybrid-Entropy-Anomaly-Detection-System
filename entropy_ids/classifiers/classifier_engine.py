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

def apply_benign_confirmation(preds: np.ndarray, df: pd.DataFrame, indices=None) -> np.ndarray:
    """
    Overrides predicted anomalies (1) to benign (0) if they meet benign profile criteria.
    """
    overridden_count = 0
    final_preds = preds.copy()
    
    iterator = range(len(final_preds)) if indices is None else indices

    for j, idx in enumerate(iterator):
        if final_preds[j] == 1:
            row = df.iloc[idx]
            if (row.get("ascii_ratio", 0) > ASCII_RATIO_MIN and 
                row.get("non_printable_ratio", 1) < NON_PRINTABLE_RATIO_MAX and 
                row.get("global_entropy", 10) < GLOBAL_ENTROPY_MAX):
                final_preds[j] = 0
                overridden_count += 1
                
    logger.info(f"Benign confirmation layer active. Overridden {overridden_count} False Positives.")
    return final_preds
