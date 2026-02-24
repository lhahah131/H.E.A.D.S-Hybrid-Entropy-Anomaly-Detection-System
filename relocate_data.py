
import os
import shutil

# Define source and destination
base_src = r"d:\malware_entropy_ml\entropy-malware-framework"
base_dst = r"d:\malware_entropy_ml\entropy_ids"

# Directories to create
dirs_to_create = [
    os.path.join(base_dst, "data", "features"),
    os.path.join(base_dst, "reports"),
    os.path.join(base_dst, "models", "saved_models")
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"Created/Verified: {d}")

# Files to copy
files_to_copy = [
    (os.path.join(base_src, "features", "expanded_dataset_v3.csv"), os.path.join(base_dst, "data", "features", "expanded_dataset_v3.csv")),
    (os.path.join(base_src, "features", "master_features.csv"), os.path.join(base_dst, "data", "features", "master_features.csv")),
    (os.path.join(base_src, "isolation_results.txt"), os.path.join(base_dst, "reports", "isolation_results.txt"))
]

for src, dst in files_to_copy:
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Source not found: {src}")
