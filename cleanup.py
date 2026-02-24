
import os
import shutil

root = r"d:\malware_entropy_ml\entropy_ids"
targets = ["classifiers", "config", "controller", "evaluation", "features", "thresholds", "logs", "models", "tests"]
main_file = os.path.join(root, "main.py")

for t in targets:
    path = os.path.join(root, t)
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")

if os.path.exists(main_file):
    try:
        os.remove(main_file)
        print(f"Removed file: {main_file}")
    except Exception as e:
        print(f"Error removing {main_file}: {e}")
