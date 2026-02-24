import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Adjusted for relocation to /research/
DATA_PATH = "../data/features/expanded_dataset_v3.csv"

# ====================
# LOAD DATA
# ====================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    DATA_PATH = "../data/features/master_features.csv"
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Ensure label exists
if "label" not in df.columns:
    def assign_label(file_id):
        fid = file_id.lower()
        if fid.startswith("benign"): return 0
        if "entropy" in fid: return 1
        if "base64" in fid: return 1
        return 0
    df["label"] = df["file_id"].apply(assign_label)

# Prepare matrices
X = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors='ignore')
y = df["label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = IsolationForest(contamination=0.12, random_state=42)
model.fit(X_scaled)

# Get anomaly scores (higher = more anomalous)
# score_samples returns opposite of decision_function (smaller is more normal)
scores = -model.score_samples(X_scaled) 

# ========================
# ROC Calculation
# ========================
fpr, tpr, thresholds = roc_curve(y, scores)
roc_auc = auc(fpr, tpr)

print("\n" + "="*40)
print(" ROC PERFORMANCE EVALUATION (RESEARCH)")
print("="*40)
print(f" AUC-ROC Score : {roc_auc:.4f}")
print("="*40 + "\n")

# ========================
# Plot ROC Curve
# ========================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Analysis')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
