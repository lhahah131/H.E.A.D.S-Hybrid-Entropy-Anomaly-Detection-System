import sys
import os
import time
import json
import logging
import datetime
import numpy as np
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import warnings
warnings.filterwarnings("ignore")
try:
    import pefile
except ImportError:
    pefile = None

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Lokasi project ---
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "app"))

# Target folder yang akan dipantau sebagai Zona Karantina / Sandbox
WATCH_DIR = os.path.join(ROOT_DIR, "data", "sandbox")

# Pastikan foldernya ada, jika tidak, otomatis dibuatkan
os.makedirs(WATCH_DIR, exist_ok=True)


class MalwareScannerHandler(FileSystemEventHandler):
    """
    Class ini akan aktif secara otomatis setiap kali ada aktivitas file
    di dalam folder yang dipantau (WATCH_DIR).
    """
    def __init__(self):
        super().__init__()
        print("[*] Memuat AI Model (Isolation Forest v1)...")
        from core.model_engine import IsolationForestEngine
        
        self.model_engine = IsolationForestEngine()
        model_path = os.path.join(ROOT_DIR, "models", "saved_models", "production", "iso_v1_production.pkl")
        
        # Fallback if old path
        if not os.path.exists(model_path):
            model_path = os.path.join(ROOT_DIR, "models", "production", "iso_v1_production.pkl")
            
        try:
            self.model_engine.load_model(model_path)
            
            # Load persisted threshold and bounds
            meta_path = model_path.replace(".pkl", "_metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.threshold = float(meta.get("trained_threshold", 0.0))
                self.calibrated_bounds = meta.get("calibrated_bounds", None)
            print(f"[+] Model dimuat sukses. Internal Threshold (IF): {self.threshold:.4f}")
            if self.calibrated_bounds:
                print(f"[+] Sensor AI Dikalibrasi otomatis! (Batas Aman: {self.calibrated_bounds['ENT_NORMAL']:.3f}, Batas Bahaya: {self.calibrated_bounds['ENT_ANOM']:.3f})")
        except Exception as e:
            print(f"[!] GAGAL memuat model: {e}")
            sys.exit(1)

    def on_created(self, event):
        # Abaikan jika yang dibuat adalah folder
        if event.is_directory:
            return
            
        filepath = event.src_path
        filename = os.path.basename(filepath)
        
        print(f"\n" + "="*60)
        print(f"🚨 [ALERT] Aktivitas Sistem Terdeteksi: {filename}")
        print("="*60)
        
        # --- File Stability Wait Loop ---
        print(f"[*] Menunggu transfer dataset/file selesai (Stabilisasi Ukuran)...")
        historical_size = -1
        retries = 0
        while retries < 15: # Maksimal tunggu 15 detik
            try:
                current_size = os.path.getsize(filepath)
                if current_size == historical_size and current_size > 0:
                    # Ukuran sudah tidak berubah selama 1 detik penuh
                    break
                historical_size = current_size
            except OSError:
                pass
            time.sleep(1)
            retries += 1
            
        if retries == 15 or historical_size == 0:
            print("[-] [TIMEOUT] File gagal dibaca utuh atau tetap 0 byte (File Kosong/Corrupt).")
            return
            
        print(f"[+] Ukuran file final stabil: {historical_size} bytes.")
        
        try:
            self.scan_file_with_heads(filepath, filename)
        except Exception as e:
            print(f"[ERROR] Gagal melakukan scanning: {e}")

    def _extract_base_features(self, filepath):
        """Mengekstrak fitur entropi dan statistik byte persis seperti sistem H.E.A.D.S."""
        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except:
            return None
            
        n = len(data)
        if n == 0:
            return None
            
        # Global Entropy
        counts = np.bincount(list(data), minlength=256)
        probs  = counts[counts > 0] / n
        global_entropy = float(-np.sum(probs * np.log2(probs)))
        
        # Block Entropy
        block_size = 256
        entropies = []
        for i in range(0, n, block_size):
            block = data[i:i + block_size]
            if len(block) > 4:
                b_counts = np.bincount(list(block), minlength=256)
                b_probs = b_counts[b_counts > 0] / len(block)
                entropies.append(float(-np.sum(b_probs * np.log2(b_probs))))
                
        b_entropies_arr = np.array(entropies) if entropies else np.array([0.0])
        block_mean = float(b_entropies_arr.mean())
        block_std = float(b_entropies_arr.std())
        
        # Ratios
        ascii_count = sum(1 for b in data if 32 <= b <= 126)
        non_printable_count = sum(1 for b in data if b < 32 or b > 126)
        ascii_ratio = ascii_count / n
        non_printable_ratio = non_printable_count / n
        
        # Byte Stats
        byte_arr = np.array(list(data), dtype=np.float64)
        byte_mean = float(byte_arr.mean())
        byte_std = float(byte_arr.std())
        
        byte_skew = float(np.mean(((byte_arr - byte_mean) / (byte_std + 1e-9)) ** 3))
        return {
            "global_entropy": global_entropy,
            "block_mean_entropy": block_mean,
            "block_std_entropy": block_std,
            "non_printable_ratio": non_printable_ratio,
            "ascii_ratio": ascii_ratio,
            "byte_mean": byte_mean,
            "byte_std": byte_std,
            "byte_skewness": byte_skew
        }

    def _extract_advanced_features(self, filepath):
        adv = {
            "is_executable": 0, "num_sections": 0, "suspicious_api_count": 0,
            "has_high_entropy_section": 0, "suspicious_string_count": 0
        }
        
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            suspicious_keywords = [b"http://", b"https://", b"powershell", b"cmd.exe", b"WScript.Shell"]
            adv["suspicious_string_count"] = sum(data.count(kw) for kw in suspicious_keywords)
        except:
            pass
            
        if pefile is not None:
            try:
                pe = pefile.PE(filepath)
                adv["is_executable"] = 1
                adv["num_sections"] = len(pe.sections)
                
                for section in pe.sections:
                    if section.get_entropy() > 7.5:
                        adv["has_high_entropy_section"] = 1
                        break
                        
                bad_apis = [b"VirtualAlloc", b"CreateRemoteThread", b"InternetOpen", b"WriteProcessMemory"]
                api_count = 0
                if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in entry.imports:
                            if imp.name and any(bad in imp.name for bad in bad_apis):
                                api_count += 1
                adv["suspicious_api_count"] = api_count
            except:
                pass
        return adv

    def scan_file_with_heads(self, filepath, filename):
        print(f"[*] Memulai Mesin Ekstraksi Fitur...")
        
        base_feats = self._extract_base_features(filepath)
        if not base_feats:
            print("[-] File kosong atau tidak dapat dibaca. Mengabaikan file.")
            return
            
        adv_feats = self._extract_advanced_features(filepath)
        merged_feats = {**base_feats, **adv_feats}
            
        # Membuat DataFrame 1 baris
        df_raw = pd.DataFrame([merged_feats])
        
        # Menerapkan feature engineering (turunan rasio)
        df_feat = df_raw.copy()
        df_feat["entropy_x_nonprint"] = df_feat["global_entropy"] * df_feat["non_printable_ratio"]
        df_feat["entropy_div_ascii"] = df_feat["global_entropy"] / df_feat["ascii_ratio"].replace(0, 0.0001)
        df_feat["bytestd_div_bytemean"] = df_feat["byte_std"] / df_feat["byte_mean"].replace(0, 0.0001)
        
        # Ensure column order matches training!
        feature_cols = [
            "global_entropy", "block_mean_entropy", "block_std_entropy",
            "non_printable_ratio", "ascii_ratio", 
            "byte_mean", "byte_std", "byte_skewness",
            "entropy_x_nonprint", "entropy_div_ascii", "bytestd_div_bytemean",
            "is_executable", "num_sections", "suspicious_api_count", 
            "has_high_entropy_section", "suspicious_string_count"
        ]
        
        X = df_feat[feature_cols].values
        
        print(f"    - Global Entropy     : {df_feat['global_entropy'][0]:.4f}")
        print(f"    - Non-Printable Ratio: {df_feat['non_printable_ratio'][0]:.4f}")
        print(f"    - ASCII Ratio        : {df_feat['ascii_ratio'][0]:.4f}")
        
        # 1. Isolation Forest Prediction
        print(f"[*] Eksekusi Tahap 1: Isolation Forest Scoring...")
        raw_scores = self.model_engine.get_scores(X)
        pred_if = (raw_scores < self.threshold).astype(int)
        
        # 2. HWCL Engine Validation
        from core.hwcl_engine import apply_hwcl_confirmation
        
        print(f"[*] Eksekusi Tahap 2: Decision Refinement (HWCL v2.2)...")
        # Ingat, optimal cumulative_threshold produksi adalah 0.35!
        final_preds = apply_hwcl_confirmation(
            pred_if, 
            df_feat, 
            raw_scores=raw_scores, 
            cumulative_threshold=0.35,
            calibrated_bounds=getattr(self, "calibrated_bounds", None)
        )
        
        final_verdict = final_preds[0]
        
        # 3. EXPLAINABLE AI (XAI) ALGORITHM
        entropy_val = df_feat['global_entropy'][0]
        np_val = df_feat['non_printable_ratio'][0]
        
        batas_aman = self.calibrated_bounds["ENT_NORMAL"] if hasattr(self, "calibrated_bounds") and self.calibrated_bounds else 4.75
        
        # HASIL FINAL
        print("\n" + "=" * 70)
        if final_verdict == 1:
            print(f" ❌ [DITOLAK] Sistem keamanan memblokir file ini: {filename}")
            print(f" 📝 [ALASAN]  Sistem kecerdasan buatan menyadari ada yang janggal:")
            
            if entropy_val > batas_aman:
                print(f"              - Isi file ini terlihat seperti 'sandi acak' yang sangat padat.")
                print(f"                Biasanya, virus atau ransomware menyamar dengan cara")
                print(f"                seperti ini agar wujud aslinya tidak ketahuan.")
            
            if np_val > 0.08:
                print(f"              - Kami juga menemukan banyak sekali karakter 'aneh'")
                print(f"                yang tidak wajar ada di dalam dokumen atau aplikasi normal.")
            
            if entropy_val <= batas_aman and np_val <= 0.08:
                 print(f"              - Walau sekilas tampak wajar, struktur di dalam file ini")
                 print(f"                dicurigai oleh lapisan keamanan kami karena mirip dengan virus.")
                 
            # Simulasi karantina:
            # os.rename(filepath, filepath + ".quarantine")
        else:
            print(f" ✅ [DITERIMA] File ini aman: {filename}")
            if entropy_val > 5.0:
                print(f" 📝 [ALASAN]  Walau isinya agak padat, namun susunannya masih masuk akal")
                print(f"              dan tidak ada tanda-tanda ancaman.")
            else:
                print(f" 📝 [ALASAN]  Semua terlihat bersih dan wajar seperti file pada umumnya.")
        print("=" * 70 + "\n")
        
        # --- MENYIMPAN RIWAYAT KE LOG FILE ---
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_text = "DIBLOKIR (MALWARE)" if final_verdict == 1 else "DITERIMA (AMAN)"
            log_entry = f"[{timestamp}] FILE: {filename} | STATUS: {status_text} | ENTROPI: {entropy_val:.3f} | NON-PRINTABLE: {np_val*100:.1f}%\n"
            
            log_dir = os.path.join(ROOT_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "scan_history.log"), "a", encoding="utf-8") as lf:
                lf.write(log_entry)
        except Exception as e:
            print(f"[-] Gagal menulis riwayat ke file log: {e}")


def start_watcher():
    event_handler = MalwareScannerHandler()
    observer = Observer()
    
    # Memasang mata-mata di folder sandbox
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    
    print("=" * 60)
    print(" 🛡️ H.E.A.D.S REAL-TIME SCANNER AKTIF 🛡️")
    print(f" Menunggu file baru di: {WATCH_DIR}")
    print(" Tekan Ctrl+C untuk menghentikan program.")
    print("=" * 60)
    
    try:
        # Biarkan program terus hidup dan berputar di latar belakang
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Program dihentikan pengguna. Mematikan scanner...")
        observer.stop()
        
    observer.join()

if __name__ == "__main__":
    start_watcher()
