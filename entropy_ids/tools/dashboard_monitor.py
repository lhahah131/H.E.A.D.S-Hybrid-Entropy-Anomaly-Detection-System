import os
import json
import time

def read_json_meta(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prod_meta_path = os.path.join(base_dir, 'models', 'production', 'v1.0', 'isolation_forest_v1.0_metadata.json')
    inf_meta_path = os.path.join(base_dir, 'logs', 'metadata_inference_adaptive.json')

    while True:
        clear_screen()
        
        # Ambil data terbaru dari disket
        prod = read_json_meta(prod_meta_path)
        inf = read_json_meta(inf_meta_path)

        print("================================================================")
        print(" 🛰️ STATUS DASHBOARD: HYBRID ANOMALY DETECTION [LIVE] 🛰️")
        print("================================================================")

        # Bagian Status Produksi Model
        print("\n[ 🔒 INTI PRODUCTION V1.0 ]")
        if prod:
            print(f"  > Versi Model     : v1.0 (Frozen)")
            print(f"  > Dibekukan Pada  : {prod.get('timestamp')}")
            print(f"  > Mode Threshold  : {prod.get('mode', 'N/A').upper()}")
            print(f"  > Locked Threshold: {prod.get('trained_threshold', 0):.4f} (Pct: {prod.get('percentile')})")
        else:
            print("  [PERINGATAN] Data meta produksi v1.0 tidak ditemukan.")
            print("  Silakan jalankan `python tools/freeze_v1.py` terlebih dahulu.")

        # Bagian Metrik Terdeteksi
        print("\n[ 📊 HASIL INFERENCE TERAKHIR ]")
        if inf:
            m = inf.get('metrics', {})
            print(f"  > Last Run    : {inf.get('timestamp')}")
            print(f"  > Action      : {inf.get('action').upper()}")
            print("-" * 50)
            print(f"  > F1 Score    : {m.get('f1', 0):.4f}")
            print(f"  > Precision   : {m.get('precision', 0):.4f}")
            print(f"  > Recall      : {m.get('recall', 0):.4f}")
            print(f"  > Alarm Palsu : {m.get('fp', 0)} kasus (Target: < 5)")
            print(f"  > Terlewat    : {m.get('fn', 0)} kasus")
            
            # Health Check Model
            if m.get('fp', 100) > 5:
                print("  [⚠️ STATUS: WARNING] False Positives tinggi!")
            else:
                print("  [✅ STATUS: AMAN] Performa Model Lulus Predikat Produksi.")
        else:
            print("  Belum ada log tes inference yang terbaca.")

        print("\n================================================================")
        print(" [CTRL+C] untuk keluar dari monitor.")
        print(" Menarik ulang data dalam 5 detik...")
        
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitor dihentikan.")
            break

if __name__ == '__main__':
    main()
