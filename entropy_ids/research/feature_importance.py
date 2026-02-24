import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Adjusted for relocation to /research/
DATA_PATH = "../data/features/expanded_dataset_v3.csv"

# ======================
# LOAD DATA
# ======================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    # Attempt fallback to old location for convenience
    DATA_PATH = "../data/features/master_features.csv"
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found in ../data/features/. Make sure you run this from the research/ folder.")
        exit(1)

file_ids = df["file_id"]
# Drop non-numeric columns for modeling
X = df.select_dtypes(include=[np.number])
if "label" in X.columns:
    X = X.drop(columns=["label"])

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# ISOLATION FOREST 
# ======================
# Using Isolation Forest labels as a proxy for "anomaly" vs "normal"
iso = IsolationForest(
    n_estimators=100,
    contamination=0.2,
    random_state=42
)

iso.fit(X_scaled)
predictions = iso.predict(X_scaled)

# Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
# We are trying to find features that distinguish the anomalies.
y = np.where(predictions == -1, 1, 0)

# ======================
# RANDOM FOREST (FEATURE IMPORTANCE)
# ======================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_scaled, y)

importances = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance Ranking ===\n")
print(feature_importance_df.head(20))
