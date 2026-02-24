from sys import meta_path
from sqlite3.dbapi2 import Timestamp
from core.feature_engine import engineer_features
import pandas as pd
import logging
import os
import json
import datetime

from core.feature_engine import feature_engine, extract_features_and_labels
from core.model_engine import BaseAnomalyModel
from core.threshold_engine import BaseThreshold
from core.classifier_engine import classify_predictions,apply_benign_confirmation
from core.evaluator_engine import evaluator_metrics
from config.settings import LOG_DIR

logger = logging.getLogger(__name__)

class HybridController:
    """Orchestrates the entire ML pipeline from feature loading to evaluation using.
       includes a Graceful Fail-Safe Recovery system to prevent production crashes.
    """
    def __init__(self,data_path:str,model: BaseAnomalyModel,threshold_engine:BaseThreshold):
        self.data_path = data_path
        self.model = model
        self.threshold_engine = threshold_engine
        self.version = "1.0.0"
        
        #--- fail-safe attributes
        self.system_mode = "NORMAL"
        self.model_loaded = True
        self.recovery_attempted = False
        self.last_save_path = None # persist model path

    def run(self,action:str, mode: str, model_save_path: str = None):
        logger.info(f"-- Starting Pipeline Exectution | Action:{action.upper()} | Mode: {mode.upper()} ---")

        #save path for potentia recovery
        if model_save_path:
            self._model_save_path = model_save_path
        
        #load path for potential recovery
        df_feats = engineer_features(df_raw)
        x, y_true, df_processed = extract_features_and_labels(df_feats)

        if action == "train":
            self.model.train(x)
            if model_save_path:
                self.model.save(model_save_path)
            self.model_loaded = True

        elif action =="interface":
            if model_save_path and getattr(self.model, "is_fitted", False) is False:
                try:
                    self.model.load(model_save_path)
                    self.model_loaded = True
                except Exception as e:
                    self._attempt_model_recovery(e)
                elif not model_save_path:
                    logger.warning("Model_save_path not provide. Depending on pre-loaded state")
                else:
                    logger.warning("Unknown action: {action}")
            # ===========================================
            # SAFE_LOG_ONLY FAST-PATH (Bypass ai Scoring)
            #============================================
            if self.system_mode == "SAVE_LOG_ONLY":
                logger.warning("Excuting in SAVE_LOG_ONLY mode. Bypassing anomaly detection")
                return self._generate_safe_log_only_response(len(x))

            #===========================================
            # NORMAL INFERENCE / SCORING PATH
            #===========================================
            try:
                raw_scores = self.model.get_scores(x)
            except Exception as e:

                #Error encounteres during scoring; trigger recovery
                self.attempt_model_recovery(e)

            # RE-check mode immediately after recovery attempt
            if self.system_mode == "SAVE_LOG_ONLY":
                return self._generate_safe_mode_response(len(x))
            else:
                # Recovery seceeded, entry scoring
                raw_scores = self.model.get_scores(x)

            #Determine threshold
            threshold, scaler_roc = self.threshold_engine.compute(raw_scores,y_true)
            logger.info(f"Final threshold: {threshold:.4f}")

            # Calculate auc scores format
            if scaler_roc is None:
                anomaly_scores_for_auc = -raw_scores
            else:
                anomaly_scores_for_auc = 1 - scaler_roc.transform(raw_scores.reshape(-1, 1)).flatten()

            # Classification
            preds = classify_predictions(raw_scores, threshold, mode, scaler_roc)

            #Benign Confirmation Layer
            final_preds = apply_benign_confirmation(preds,df_processed)

            #Evaluation
            metrics = evaluator_metrics(y_true, final_preds, anomaly_scores_for_auc)

            #Save predictions & metadata
            self._save_results(f"{action}_{mode}", df_processed, final_preds, raw_scores, threshold, metrics)         
            return metrics, threshold

    def _attempt_model_recovery(self, error: Exception):
        """Graceful recovery triggered ONLY open runtime errors,
           Prevents infinite looping of reloads.
        """
        timestamp = datetime.datetime.now().isoformat()
        logger.warning(f"[{timestamp}] ML ERROR ENCOUNTERED: {type(error).__name__} - {str(error)}")

        if self.recovery_attempted:
            logger.warning("Recovery attempt already attempted previously. Enforcing SAVE_LOG_ONLY mode")
            self.system_mode = "SAVE_LOG_ONLY"
            self.model_loaded = False
            return

            logger.info("Attempting single emergency model reload")
            try:
                if not self.model_save_path:
                    raise ValueError("No model path configured for recovery.")

                self.model.load(self._model_save_path)

                # Recovery Successful 
                self.system_mode = "NORMAL"
                self.model_loaded = True
                logger.info(f"EMERGENCY RECOVERY SUCCESSFUL. Returning to NORMAL mode.")

                except Exception as recovery_error:
                    self.system_mode = "SAVE_LOG_ONLY"
                    self.model_loaded = False
                    logger.error(f"EMERGENCY RECOVERY FAILED: {type(recovery_error).}Switching to SAVE_LOG_ONLY.")
                    
                    finally:

                         # Mark recovery as attempted regardless of outcome to prevent infinite loops
                        self.recovery_attempted = True
                        
                        #log final telemetry footprint for alerting systems
                        logger.warning(f"SYSTEM STATE UPDATE -> Mode:" {self.system_mode}
                                        "Model Loaded: " {self.model_loaded}
                                        "Recovery Attempted: " {self.recovery_attempted}
                                        "Last Save Path: " {self.last_save_path})
                    def _generate_safe_log_only_response(self, sample_count):
                        """
                        Structured response mapped when ai scoring is mechanisms are offline
                        """
                        response = {
                            "status": "SAVE_MODE",
                            "action": "LOG_ONLY",
                            "reason": "Model unavailable / Critical ML Error",
                            "metadata": {
                                "samples_processed": sample_count,
                                "timetamp": datetime.datetime.now().isoformat()
                            }
                        }
                        logger.info(f"Generated SAVE_MODE payload: {json.dump(response)}")
                        return response, None

                def _save_results(self, mode, df_processed, final_preds, raw_scores, thresholed, metrics):
                    output_df = df_processed.copy()
                    output_df["pred_anomaly"] = final_preds
                    output_df["raw_score"] = raw_scores

                    os.makedirs(LOG_DIR, exist_ok = True)
                    #predictions
                    out_path = os.path.join(LOG_DIR, f"predictions_{mode}.csv")
                    output_df.to_csv(out_path, index = False)
                    logger.info(f"Saved mode {mode.upper()} predictions to {out_path}")

                    #Versioning Model Metadata
                    metadata = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "version": self.version,
                        "mode": mode,
                        "threshold": thresholed,
                        "metrics": metrics
                    }

                    meta_path = os.path.join(LOG_DIR, f"metadata_{mode}.json")
                    with open(meta_path, "w") as f:
                        json.dump(metadata, f, indent = 4)
                     logger.info(f"Saved execution metadata to {meta_path}")