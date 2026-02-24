import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Adjusted for relocation to /research/
DATA_PATH = "../data/features/expanded_dataset_v3.csv"

# ======================
# LOAD DATA
# ======================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    DATA_PATH = "../data/features/master_features.csv"
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found in ../data/features/. Make sure you run this from the research/ folder.")
        exit(1)

file_ids = df["file_id"]
X = df.select_dtypes(include=[np.number])
if "label" in X.columns:
    X = X.drop(columns=["label"])

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# KMEANS
# ======================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

print("\nCluster Assignments (First 20):")
for file_id, cluster_id in zip(file_ids[:20], clusters[:20]):
    print(f"{file_id} -> Cluster {cluster_id}")

# =======================
# PCA FOR VISUALIZATION
# =======================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')

# Labeling first 30 points to avoid clutter
for i, txt in enumerate(file_ids[:30]):
    plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

plt.title("K-Means Clustering Visualization (Research Phase)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.colorbar(scatter, label="Cluster ID")
# Ensure directory for plots exists if we wanted to save, but here we just show/comment
# plt.savefig("../reports/kmeans_clusters.png")
plt.show()
