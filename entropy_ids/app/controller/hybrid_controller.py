import pandas as pd
import numpy as np
import logging
import os
import json
import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from core.feature_engine import engineer_features, extract_features_and_labels
from core.model_engine import BaseAnomalyModel
from core.threshold_engine import BaseThreshold
from core.classifier_engine import classify_predictions, apply_benign_confirmation
from core.evaluation_engine import evaluate_metrics
from config.settings import LOG_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE, CV_FOLDS, PERCENTILE_VALUE

logger = logging.getLogger(__name__)

class HybridController:
    """
    Orchestrates the entire ML pipeline from feature loading to evaluation using Injection.
    Includes a Graceful Fail-Safe Recovery system and Production-Grade Evaluation Pipeline.
    """
    def __init__(self, data_path: str, model: BaseAnomalyModel, threshold_engine: BaseThreshold):
        self.data_path = data_path
        self.model = model
        self.threshold_engine = threshold_engine
        self.version = "1.1.0"
        
        # --- Fail-Safe State Attributes ---
        self.system_mode = "NORMAL"
        self.model_loaded = True
        self.recovery_attempted = False
        self._model_save_path = None # Persisted for recovery use
        
    def run(self, action: str, mode: str, model_save_path: str = None):
        logger.info(f"--- Starting Pipeline Execution | Action: {action.upper()} | Mode: {mode.upper()} ---")
        
        # Save path for potential recovery
        if model_save_path:
            self._model_save_path = model_save_path
            
        # Load data
        df_raw = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df_raw)} records from {self.data_path}")
        
        # Feature Engineering
        df_feats = engineer_features(df_raw)
        X, y_true, df_processed = extract_features_and_labels(df_feats)
        
        train_size = len(X)
        test_size = len(X)
        
        metrics = None
        threshold = 0.0

        if action == "train":
            if len(X) < 30:
                logger.warning("Small dataset detected. Metrics may be unstable.")
                
            # Stratify only if we have more than 1 class
            stratify_col = y_true if len(np.unique(y_true)) > 1 else None
            
            X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
                X, y_true, df_processed, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=stratify_col
            )
            
            train_size = len(X_train)
            test_size = len(X_test)
            logger.info(f"Splitting data -> Train: {train_size}, Test: {test_size}")
            
            # Train model ONLY on X_train
            self.model.train(X_train)
            if model_save_path:
                self.model.save_model(model_save_path)
            self.model_loaded = True
            
            # Evaluate ONLY on X_test to prevent leakage
            metrics, threshold = self._evaluate_and_log(
                X_test, y_test, df_test, mode, action, train_size, test_size
            )
            return metrics, threshold

        elif action == "inference":
            if model_save_path and getattr(self.model, 'is_fitted', False) is False:
                try:
                    self.model.load_model(model_save_path)
                    self.model_loaded = True
                except Exception as e:
                    self._attempt_model_recovery(e)
            elif not model_save_path:
                logger.warning("model_save_path not provided. Depending on pre-loaded state.")
                
            metrics, threshold = self._evaluate_and_log(
                X, y_true, df_processed, mode, action, 0, test_size
            )
            return metrics, threshold
            
        elif action == "cv":
            if len(X) < 30:
                logger.warning("Small dataset detected. CV Metrics may be unstable.")
            
            logger.info(f"Starting {CV_FOLDS}-Fold Cross Validation...")
            skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            cv_metrics = {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'fp': [], 'fn': []}
            fold_thresholds = []
            
            for train_idx, test_idx in skf.split(X, y_true):
                X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                y_train_cv, y_test_cv = y_true[train_idx], y_true[test_idx]
                df_test_cv = df_processed.iloc[test_idx].reset_index(drop=True)
                
                # Train fold model
                self.model.train(X_train_cv)
                
                # Infer on test
                raw_scores = self.model.get_scores(X_test_cv)
                t, scaler_roc = self.threshold_engine.compute(raw_scores, y_test_cv)
                fold_thresholds.append(t)
                
                if scaler_roc is None:
                    anomaly_scores_for_auc = -raw_scores
                else:
                    anomaly_scores_for_auc = 1 - scaler_roc.transform(raw_scores.reshape(-1, 1)).flatten()
                    
                preds = classify_predictions(raw_scores, t, mode, scaler_roc)
                final_preds = apply_benign_confirmation(preds, df_test_cv)
                
                m = evaluate_metrics(y_test_cv, final_preds, anomaly_scores_for_auc)
                for k in cv_metrics:
                    cv_metrics[k].append(m[k])
                    
            avg_metrics = {k: float(np.mean(v)) for k, v in cv_metrics.items()}
            threshold = float(np.mean(fold_thresholds))
            
            logger.info("CV completed. Average metrics generated.")
            return avg_metrics, threshold

        else:
            raise ValueError(f"Unknown action: {action}")

    def _evaluate_and_log(self, X, y_true, df_processed, mode, action, train_size, test_size):
        # -------------------------------------------------------------
        # SAFE_LOG_ONLY FAST-PATH (Bypass AI Scoring)
        # -------------------------------------------------------------
        if self.system_mode == "SAFE_LOG_ONLY":
            logger.warning("Executing in SAFE_LOG_ONLY mode. Bypassing anomaly detection.")
            return self._generate_safe_mode_response(len(X))
            
        # -------------------------------------------------------------
        # NORMAL INFERENCE / SCORING PATH
        # -------------------------------------------------------------
        try:
            raw_scores = self.model.get_scores(X)
        except Exception as e:
            # Error encountered during scoring; trigger fail-safe
            self._attempt_model_recovery(e)
            
            # Re-check mode immediately after recovery attempt
            if self.system_mode == "SAFE_LOG_ONLY":
                return self._generate_safe_mode_response(len(X))
            else:
                # Recovery suceeded, retry scoring
                raw_scores = self.model.get_scores(X)
        
        # Determine threshold
        if action == "train":
            threshold, scaler_roc = self.threshold_engine.compute(raw_scores, y_true)
            logger.info(f"Final threshold applied: {threshold:.4f}")
            self._save_trained_threshold(threshold, mode)
        elif action == "inference":
            scaler_roc = None
            original_threshold = self._load_trained_threshold()
            threshold = original_threshold
            logger.info("Loaded persisted threshold from training metadata.")
            logger.info(f"Final threshold applied: {threshold:.4f}")
        
        # Calculate auc scores format
        if scaler_roc is None:
            anomaly_scores_for_auc = -raw_scores
        else:
            anomaly_scores_for_auc = 1 - scaler_roc.transform(raw_scores.reshape(-1, 1)).flatten()
            
        # Classification
        preds = classify_predictions(raw_scores, threshold, mode, scaler_roc)
        
        # Benign Confirmation Layer
        final_preds = apply_benign_confirmation(preds, df_processed)
        
        # Evaluation
        metrics = evaluate_metrics(y_true, final_preds, anomaly_scores_for_auc)
        
        # Save predictions & metadata
        self._save_results(action, mode, df_processed, final_preds, raw_scores, threshold, metrics, train_size, test_size)
        
        return metrics, threshold

    def _attempt_model_recovery(self, error: Exception):
        """
        Graceful recovery protocol triggered ONLY upon runtime errors.
        Prevents infinite looping of reloads.
        """
        timestamp = datetime.datetime.now().isoformat()
        logger.warning(f"[{timestamp}] ML ERROR ENCOUNTERED: {type(error).__name__} - {str(error)}")
        
        if self.recovery_attempted:
            logger.error("Recovery was already attempted previously. Enforcing SAFE_LOG_ONLY mode.")
            self.system_mode = "SAFE_LOG_ONLY"
            self.model_loaded = False
            return
            
        logger.info("Attempting single emergency model reload...")
        
        try:
            if not self._model_save_path:
                raise ValueError("No model path configured for recovery.")
            
            self.model.load_model(self._model_save_path)
            
            # Recovery Successful
            self.system_mode = "NORMAL"
            self.model_loaded = True
            logger.info("EMERGENCY RECOVERY SUCCESSFUL. Returning to NORMAL mode.")
            
        except Exception as recovery_error:
            # Recovery Failed
            self.system_mode = "SAFE_LOG_ONLY"
            self.model_loaded = False
            logger.error(f"EMERGENCY RECOVERY FAILED: {str(recovery_error)}. Switching to SAFE_LOG_ONLY.")
            
        finally:
            # Mark recovery as attempted regardless of outcome to prevent infinite loops
            self.recovery_attempted = True
            
            # Log final telemetry footprint for alerting systems
            logger.warning(f"SYSTEM STATE UPDATE -> Mode: {self.system_mode}, "
                           f"Model Loaded: {self.model_loaded}, "
                           f"Recovery Attempted: {self.recovery_attempted}")

    def _generate_safe_mode_response(self, sample_count):
        """
        Structured response mapped when AI scoring mechanisms are offline.
        """
        response = {
            "status": "SAFE_MODE",
            "action": "LOG_ONLY",
            "reason": "Model unavailable / Critical ML Error",
            "metadata": {
                "samples_processed": sample_count,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
        logger.info(f"Generated SAFE_MODE payload: {json.dumps(response)}")
        return response, None

    def _save_results(self, action, mode, df_processed, final_preds, raw_scores, threshold, metrics, train_size, test_size):
        output_df = df_processed.copy()
        output_df["pred_anomaly"] = final_preds
        output_df["raw_score"] = raw_scores
        
        os.makedirs(LOG_DIR, exist_ok=True)
        # Predictions
        out_path = os.path.join(LOG_DIR, f"predictions_{action}_{mode}.csv")
        output_df.to_csv(out_path, index=False)
        logger.info(f"Saved mode {action.upper()} predictions to {out_path}")
        
        # Versioning Model Metadata with Split Tracking
        metadata = {
            "train_size": train_size,
            "test_size": test_size,
            "split_ratio": float(TRAIN_TEST_SPLIT),
            "random_state": RANDOM_STATE,
            "threshold": threshold,
            "mode": mode,
            "action": action,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "version": self.version
        }
        meta_path = os.path.join(LOG_DIR, f"metadata_{action}_{mode}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved execution metadata to {meta_path}")

    def _save_trained_threshold(self, threshold: float, mode: str):
        if not self._model_save_path:
            return
        meta_path = self._model_save_path.replace(".pkl", "_metadata.json")
        payload = {
            "trained_threshold": threshold,
            "mode": mode,
            "percentile": PERCENTILE_VALUE,
            "timestamp": datetime.datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved trained threshold metadata to {meta_path}")

    def _load_trained_threshold(self) -> float:
        if not self._model_save_path:
            raise ValueError("model_save_path is required to load trained threshold.")
        meta_path = self._model_save_path.replace(".pkl", "_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Threshold metadata not found at {meta_path}. Please TRAIN model first.")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return float(metadata.get("trained_threshold", 0.0))
