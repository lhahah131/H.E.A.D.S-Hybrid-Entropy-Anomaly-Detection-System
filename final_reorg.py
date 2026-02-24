
import os
import shutil

# Paths
ids_root = r"d:\malware_entropy_ml\entropy_ids"
framework_root = r"d:\malware_entropy_ml\entropy-malware-framework"

# 1. Ensure new structure exists
new_dirs = [
    os.path.join(ids_root, "data", "features"),
    os.path.join(ids_root, "reports"),
    os.path.join(ids_root, "models", "saved_models"),
    os.path.join(ids_root, "logs")
]

for d in new_dirs:
    os.makedirs(d, exist_ok=True)

# 2. Rescue important files from Framework
rescue_map = [
    (os.path.join(framework_root, "isolation_results.txt"), os.path.join(ids_root, "reports", "isolation_results.txt")),
    (os.path.join(framework_root, "requirements.txt"), os.path.join(ids_root, "requirements.txt")),
    (os.path.join(framework_root, "README.md"), os.path.join(ids_root, "README_old.md"))
]

for src, dst in rescue_map:
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            print(f"Rescued: {src} -> {dst}")
        except Exception as e:
            print(f"Error rescuing {src}: {e}")

# 3. List of redundant folders in entropy_ids root to DELETE
# These are redundant because they are now in app/ or data/ or research/
redundant_folders = [
    "classifiers", "config", "controller", "evaluation", "features", 
    "thresholds", "logs", "models", "tests"
]
# Wait, I should BE CAREFUL. 
# app/ logs should be kept but root level logs deleted.
# Actually, let's look at logs again. 
# Root/logs has the system.log from my app runs.
# BUT settings.py says LOG_DIR = os.path.join(BASE_DIR, "logs"), which points to root/logs.
# So I should actually KEEP root/logs but empty the others.

folders_to_delete = [
    "classifiers", "config", "controller", "evaluation", "features", 
    "thresholds", "models", "tests"
]

for f in folders_to_delete:
    path = os.path.join(ids_root, f)
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Deleted redundant folder: {path}")
        except Exception as e:
            print(f"Could not delete {path}: {e}")

# 4. Delete redundant files in root
files_to_delete = ["main.py", "relocate_data.py", "cleanup.py"]
for f in files_to_delete:
    path = os.path.join(ids_root, f)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Deleted redundant file: {path}")
        except Exception as e:
            print(f"Could not delete {path}: {e}")

print("Cleanup complete. Structure is now: app/, research/, data/, reports/, logs/.")
