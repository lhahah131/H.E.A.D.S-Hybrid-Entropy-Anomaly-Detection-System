
import shutil
import os

src = r"d:\malware_entropy_ml\archive\legacy_v1\entropy-malware-framework\features\expanded_dataset_v3.csv"
dst = r"d:\malware_entropy_ml\entropy_ids\data\features\expanded_dataset_v3.csv"

if os.path.exists(src):
    shutil.copy2(src, dst)
    print("File successfully restored.")
else:
    print("Source file not found in archive.")
