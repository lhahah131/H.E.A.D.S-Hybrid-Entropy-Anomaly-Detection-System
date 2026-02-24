"""
=============================================================
  ML OPTIMIZATION ENGINE — Adaptive Percentile Threshold
  Senior ML Optimization Engineer Edition
  Steps: 5-Fold CV | Tuned IF | Adaptive Percentile | Features
=============================================================
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

# Adjusted for relocation to /research/
DATA_DIR = "../data/features/"
FILES = ["expanded_dataset_v3.csv", "master_features.csv"]

# ============================================================
# LOAD DATA
# ============================================================
df = None
for f in FILES:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[Info] Loaded {f}")
        break

if df is None:
    print("Error: Could not find dataset files in ../data/features/.")
    exit(1)

file_ids = df["file_id"].values

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("=" * 65)
print("  ML OPTIMIZATION ENGINE — Isolation Forest v2")
print("=" * 65)
print("\n[Feature Engineering]")

# Interaction and structural features
if "text_length" not in df.columns:
    df["text_length"] = df["file_size"] / (df["line_count"].replace(0, 1))

if "entropy_x_nonprint" not in df.columns:
    df["entropy_x_nonprint"] = df["global_entropy"] * df["non_printable_ratio"]

if "entropy_x_digit" not in df.columns:
    df["entropy_x_digit"] = df["global_entropy"] * df["digit_ratio"]

feature_cols = [
    "global_entropy", "block_mean_entropy", "block_std_entropy", "block_entropy_range",
    "digit_ratio", "non_printable_ratio", "uppercase_ratio", "symbol_ratio",
    "byte_skewness", "byte_kurtosis", "avg_line_length", "text_length",
    "entropy_x_nonprint", "entropy_x_digit"
]

feature_cols = [c for c in feature_cols if c in df.columns]
print(f"  Features used: {len(feature_cols)}")

X = df[feature_cols].values

# ============================================================
# GROUND TRUTH LABELS
# ============================================================
if "label" in df.columns:
    y_true = df["label"].fillna(0).astype(int).values
else:
    # Use benign_ prefix as proxy for label if not explicitly present
    y_true = np.where(pd.Series(file_ids).str.startswith("benign_"), 0, 1).astype(int)

n_samples = len(X)
n_anomaly = int(y_true.sum())
actual_contamination = round(min(0.5, max(0.01, n_anomaly / n_samples)), 4)

print(f"  Samples: {n_samples} | Anomalies: {n_anomaly}")

# ============================================================
# MULTI-PERCENTILE EXPERIMENT
# ============================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

percentiles_to_test = [55, 60, 65, 70, 75]
results = []

for P in percentiles_to_test:
    cv_f1, cv_prec, cv_rec, cv_fp, cv_fn = [], [], [], [], []
    
    for train_idx, test_idx in skf.split(X, y_true):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        iso = IsolationForest(n_estimators=300, contamination=actual_contamination, random_state=42, n_jobs=-1)
        iso.fit(X_train_sc)

        scores_test = iso.decision_function(X_test_sc)
        threshold = np.percentile(scores_test, P)
        
        y_pred = (scores_test < threshold).astype(int)

        # Benign confirmation layer
        for j, global_i in enumerate(test_idx):
            if y_pred[j] == 1:
                row = df.iloc[global_i]
                if (row.get("ascii_ratio", 0) > 0.85 and 
                    row.get("non_printable_ratio", 1) < 0.05 and 
                    row.get("global_entropy", 10) < 4.8):
                    y_pred[j] = 0

        cv_f1.append(f1_score(y_test, y_pred, zero_division=0))
        cv_prec.append(precision_score(y_test, y_pred, zero_division=0))
        cv_rec.append(recall_score(y_test, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cv_fp.append(cm[0][1])
        cv_fn.append(cm[1][0])

    results.append({
        "Percentile": P,
        "Precision": np.mean(cv_prec),
        "Recall": np.mean(cv_rec),
        "F1": np.mean(cv_f1),
        "Std_F1": np.std(cv_f1),
        "FP": np.sum(cv_fp),
        "FN": np.sum(cv_fn)
    })

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "─" * 75)
print(f"{'Percentile':<12} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'FP':<5} | {'FN':<5}")
print("-" * 75)
for r in results:
    print(f"{r['Percentile']:<12} | {r['Precision']:<10.4f} | {r['Recall']:<10.4f} | {r['F1']:<10.4f} | {r['FP']:<5} | {r['FN']:<5}")

valid = [r for r in results if r["Recall"] >= 0.80]
best = sorted(valid if valid else results, key=lambda x: -x["F1"])[0]
optimal_p = best["Percentile"]

print("\nSelection: Optimal Percentile =", optimal_p)
print(f"  F1 Score: {best['F1']:.4f}")
print(f"  Recall  : {best['Recall']:.4f}")
print("=" * 65)
