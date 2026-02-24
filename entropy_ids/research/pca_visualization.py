import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Adjusted for relocation to /research/
DATA_PATH = "../data/features/expanded_dataset_v3.csv"

# ======================
# LOAD DATASET
# ======================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    DATA_PATH = "../data/features/master_features.csv"
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found in ../data/features/")
        exit(1)

# Separate metadata and features
file_ids = df["file_id"]
labels = df["label"] if "label" in df.columns else None

X = df.select_dtypes(include=[np.number])
if "label" in X.columns:
    X = X.drop(columns=["label"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot scatter points
plt.figure(figsize=(12, 9))

if labels is not None:
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label="Benign (0) vs Anomaly (1)")
else:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

# Annotate specific points to reduce clutter
for i, file_id in enumerate(file_ids[:50]):
    plt.annotate(file_id, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.6)

plt.title("PCA Dimensionality Reduction - Anomaly Space Exploration")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.savefig("../reports/pca_projection.png")
plt.show()
