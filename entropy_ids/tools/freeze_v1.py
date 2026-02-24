import os
import shutil
import datetime

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_model = os.path.join(base_dir, 'models', 'saved_models', 'iso_v2.pkl')
    source_meta = os.path.join(base_dir, 'models', 'saved_models', 'iso_v2_metadata.json')
    
    prod_dir = os.path.join(base_dir, 'models', 'production', 'v1.0')
    os.makedirs(prod_dir, exist_ok=True)
    
    dest_model = os.path.join(prod_dir, 'isolation_forest_v1.0.pkl')
    dest_meta = os.path.join(prod_dir, 'isolation_forest_v1.0_metadata.json')
    
    if os.path.exists(source_model) and os.path.exists(source_meta):
        shutil.copy2(source_model, dest_model)
        shutil.copy2(source_meta, dest_meta)
        
        print("\n" + "="*50)
        print("❄️  PRODUCTION MODEL FROZEN (V1.0)")
        print("="*50)
        print(f"Time  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model : {dest_model}")
        print(f"Meta  : {dest_meta}")
        print("="*50)
        print("Versi ini kini aman dan tidak akan tertimpa oleh eksperimen training di masa depan.\n")
    else:
        print("[ERROR] Failed to find source models in models/saved_models/. Harap pastikan model sudah di-train.")

if __name__ == '__main__':
    main()
