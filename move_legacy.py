
import os
import shutil

base_dir = r"d:\malware_entropy_ml"
archive_dir = os.path.join(base_dir, "archive", "legacy_v1")
source_dir = os.path.join(base_dir, "entropy-malware-framework")
dest_dir = os.path.join(archive_dir, "entropy-malware-framework")

if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)
    print(f"Created directory: {archive_dir}")

if os.path.exists(source_dir):
    try:
        shutil.move(source_dir, dest_dir)
        print(f"Successfully moved {source_dir} to {dest_dir}")
    except Exception as e:
        print(f"Failed to move directory: {e}")
else:
    print(f"Source directory {source_dir} already moved or does not exist.")
