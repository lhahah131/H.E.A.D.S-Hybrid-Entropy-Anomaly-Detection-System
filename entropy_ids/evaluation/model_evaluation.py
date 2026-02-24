import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, anomaly_scores=None) -> dict:
    """
    Calculates precision, recall, F1, FP, FN, and optionally ROC-AUC.
    """
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fp = int(cm[0][1])
        fn = int(cm[1][0])
    except Exception:
        fp = 0
        fn = 0
        
    auc = float('nan')
    if anomaly_scores is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, anomaly_scores)
        except Exception:
            pass

    metrics = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fp": fp,
        "fn": fn,
        "roc_auc": auc
    }
    logger.info(f"Evaluated Metrics -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, FP: {fp}, FN: {fn}")
    return metrics
