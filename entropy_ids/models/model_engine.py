import numpy as np
import logging
import joblib
import os
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config.settings import RANDOM_SEED, CONTAMINATION_RATIO, N_ESTIMATORS

logger = logging.getLogger(__name__)

class BaseAnomalyModel(ABC):
    """Abstract Base Class for Anomaly Detection Models."""
    @abstractmethod
    def train(self, X: np.ndarray):
        pass

    @abstractmethod
    def get_scores(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass

class IsolationForestEngine(BaseAnomalyModel):
    """
    Isolation Forest implementation adhering to BaseAnomalyModel.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION_RATIO,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_count = 0

    def train(self, X: np.ndarray):
        """Trains the scaler and the model."""
        logger.info(f"Training model on {X.shape[0]} samples...")
        self.feature_count = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("Model trained successfully.")

    def get_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw decision_function scores. 
        Lower scores = more anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Cannot perform inference.")
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def save_model(self, path: str):
        """Saves IsolationForest model, scaler, and metadata using joblib."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "metadata": {
                "n_estimators": N_ESTIMATORS,
                "contamination": CONTAMINATION_RATIO,
                "feature_count": self.feature_count,
            }
        }
        joblib.dump(payload, path)
        logger.info(f"Model and scaler successfully saved to {path} (Features: {self.feature_count}).")

    def load_model(self, path: str):
        """Loads IsolationForest model, scaler, and sets is_fitted property."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        logger.info(f"Loading persistent model from {path}...")
        payload = joblib.load(path)
        
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_count = payload["metadata"]["feature_count"]
        self.is_fitted = True
        
        logger.info(f"Model loaded successfully. Expected features: {self.feature_count}.")
