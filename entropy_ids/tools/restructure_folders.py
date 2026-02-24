import os
import shutil

def move_files(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    else:
        print(f"File {src} not found, skipping.")

def move_wildcard(src_dir, dest_dir, prefix):
    if os.path.exists(src_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for _, _, files in os.walk(src_dir):
            for file in files:
                if file.startswith(prefix) and file.endswith('.csv'):
                    src = os.path.join(src_dir, file)
                    dst = os.path.join(dest_dir, file)
                    shutil.move(src, dst)
                    print(f"Moved {src} to {dst}")

def main():
    base_dir = r"d:\malware_entropy_ml\entropy_ids"
    
    # Create directories
    dirs = [
        "models/production",
        "models/experimental",
        "data/archive",
        "logs/production_logs",
        "logs/experimental_logs"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        
    # Move model to production
    move_files(
        os.path.join(base_dir, "models", "saved_models", "iso_v2.pkl"),
        os.path.join(base_dir, "models", "production", "iso_v1_production.pkl")
    )
    move_files(
        os.path.join(base_dir, "models", "saved_models", "iso_v2_metadata.json"),
        os.path.join(base_dir, "models", "production", "iso_v1_production_metadata.json")
    )
    
    # Also move from production/v1.0 if it exists there
    move_files(
        os.path.join(base_dir, "models", "production", "v1.0", "isolation_forest_v1.0.pkl"),
        os.path.join(base_dir, "models", "production", "iso_v1_production.pkl")
    )
    move_files(
        os.path.join(base_dir, "models", "production", "v1.0", "isolation_forest_v1.0_metadata.json"),
        os.path.join(base_dir, "models", "production", "iso_v1_production_metadata.json")
    )

    # Move experiments
    move_wildcard(
        os.path.join(base_dir, "data", "synthetic"),
        os.path.join(base_dir, "data", "archive"),
        "expanded_dataset_v"
    )
    
    print("Cleanup successful.")

if __name__ == '__main__':
    main()
