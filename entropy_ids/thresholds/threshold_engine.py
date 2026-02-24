import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from config.settings import PERCENTILE_VALUE, STRICT_THRESHOLD_MULTIPLIER

logger = logging.getLogger(__name__)

class BaseThreshold(ABC):
    """Abstract Base Class for Threshold Calculation."""
    @abstractmethod
    def compute(self, scores: np.ndarray, y_true: np.ndarray = None):
        pass

class AdaptiveThreshold(BaseThreshold):
    """Adaptive Percentile Threshold Strategy."""
    def __init__(self, percentile: float = PERCENTILE_VALUE):
        self.percentile = percentile

    def compute(self, scores: np.ndarray, y_true: np.ndarray = None):
        threshold = np.percentile(scores, self.percentile)
        logger.info(f"Calculated Adaptive Threshold (Percentile {self.percentile}): {threshold:.4f}")
        return threshold, None

class StrictThreshold(BaseThreshold):
    """Strict ROC-Youden based Threshold Strategy."""
    def __init__(self, multiplier: float = STRICT_THRESHOLD_MULTIPLIER):
        self.multiplier = multiplier

    def compute(self, scores: np.ndarray, y_true: np.ndarray = None):
        if y_true is None:
            raise ValueError("Strict threshold requires ground truth labels (y_true).")

        mm = MinMaxScaler()
        scores_norm = mm.fit_transform(scores.reshape(-1, 1)).flatten()
        anomaly_scores = 1 - scores_norm
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
            optimal_idx = np.argmax(tpr - fpr)
            opt_thresh = thresholds[optimal_idx] * self.multiplier
            logger.info(f"Calculated Strict Threshold: {opt_thresh:.4f}")
            return opt_thresh, mm
        else:
            logger.warning("Only 1 class present. Defaulting ROC threshold to 0.5.")
            return 0.5 * self.multiplier, mm
