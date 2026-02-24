import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../app'))

from config.settings import N_ESTIMATORS, CONTAMINATION_RATIO, PERCENTILE_VALUE
from core.feature_engine import engineer_features, extract_features_and_labels
from core.classifier_engine import apply_benign_confirmation

def main():
    print("=" * 60)
    print("  🚀 PRODUCTION AUDIT CHECKLIST REPORT")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'production', 'iso_v1_production.pkl')
    meta_path = os.path.join(base_dir, 'models', 'production', 'iso_v1_production_metadata.json')
    data_path = os.path.join(base_dir, 'data', 'raw', 'master_features.csv')
    
    # 1. Consistency Model & Threshold
    print("\n[1] KONSISTENSI MODEL & THRESHOLD")
    if os.path.exists(model_path):
        payload = joblib.load(model_path)
        model = payload["model"]
        scaler = payload["scaler"]
        mod_meta = payload["metadata"]
        print(f"  [OK] Model Terbaca. Estimators: {mod_meta.get('n_estimators')} | Contamination: {mod_meta.get('contamination')}")
        if mod_meta.get('n_estimators') != N_ESTIMATORS:
            print(f"       -> WARNING: Config Settings minta {N_ESTIMATORS} tapi model diajari {mod_meta.get('n_estimators')}")
        if mod_meta.get('contamination') != CONTAMINATION_RATIO:
            print(f"       -> WARNING: Config Settings minta {CONTAMINATION_RATIO} tapi model diajari {mod_meta.get('contamination')}")
    else:
        print("  [ERROR] Model iso_v1_production.pkl tidak ditemukan.")
        return

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        trained_threshold = meta.get("trained_threshold")
        print(f"  [OK] Metadata Terbaca. Trained Threshold: {trained_threshold:.4f} | Training Percentile: {meta.get('percentile')}")
        if meta.get('percentile') != PERCENTILE_VALUE:
            print(f"       -> WARNING: Config Percentile saat ini {PERCENTILE_VALUE} tapi model ditrain dengan {meta.get('percentile')}")
    else:
        trained_threshold = None
        print("  [WARNING] Metadata threshold iso_v1_production_metadata.json tidak ditemukan.")

    # 2. Konsistensi Feature
    print("\n[2] KONSISTENSI FITUR")
    df_raw = pd.read_csv(data_path)
    df_feats = engineer_features(df_raw)
    X, y_true, df_processed = extract_features_and_labels(df_feats)
    
    print(f"  [INFO] Features extracted shape: {X.shape[1]} columns")
    print(f"  [INFO] Model expected features : {mod_meta.get('feature_count')} columns")
    if X.shape[1] == mod_meta.get('feature_count'):
        print("  [OK] Jumlah fitur MATCH.")
    else:
        print("  [ERROR] JUMLAH FITUR TIDAK MATCH!")
        
    has_nan = np.isnan(X).any()
    print(f"  [OK] Apakah ada NaN di Inference Dataset? {has_nan}")

    # 3. Distribusi Data
    print("\n[3] DISTRIBUSI DATA (DRIFT CHECK)")
    b_ent = df_processed[y_true == 0]['global_entropy']
    a_ent = df_processed[y_true == 1]['global_entropy']
    print(f"  True Benign Entropy Mean : {b_ent.mean():.4f}  (Std: {b_ent.std():.4f})")
    print(f"  True Anomaly Entropy Mean: {a_ent.mean():.4f}  (Std: {a_ent.std():.4f})")

    # 4 & 5. Evaluasi Performa Real & Alarm Rate
    print("\n[4 & 5] EVALUASI PERFORMA REAL & ALARM RATE")
    X_scaled = scaler.transform(X)
    raw_scores = model.decision_function(X_scaled)
    
    # Memastikan evaluasi menggunakan PERSISTED THRESHOLD (TIDAK BOLEH NGITUNG PERCENTILE LAGI)
    inf_threshold = trained_threshold
    preds = (raw_scores < inf_threshold).astype(int)
    
    # Apply Strict Confirmation
    final_preds = apply_benign_confirmation(preds, df_processed)
    
    alarm_rate = (final_preds.sum() / len(final_preds)) * 100
    cm = confusion_matrix(y_true, final_preds, labels=[0, 1])
    try:
        fp = cm[0][1]
        fn = cm[1][0]
    except:
        fp = fn = 0
        
    real_f1 = f1_score(y_true, final_preds, zero_division=0)
    print(f"  Inference Threshold Applied: {inf_threshold:.4f} (Menggunakan Persisted Threshold dari Training)")
    print(f"  Alarm Rate (Flagged %): {alarm_rate:.1f}% dari total dataset")
    print(f"  Real F1 Score : {real_f1:.4f}")
    print(f"  Precision     : {precision_score(y_true, final_preds, zero_division=0):.4f}")
    print(f"  Recall        : {recall_score(y_true, final_preds, zero_division=0):.4f}")
    print(f"  False Positives (Aman dituduh Malware) : {fp}")
    print(f"  False Negatives (Malware lolos)        : {fn}")
    if fp == 0:
        print("  [OK] Keren! Alarm Palsu = 0.")

    # 6. Review Sample Anomaly
    print("\n[6] REVIEW SAMPLE ANOMALY (Mencurigakan)")
    anomaly_indices = np.where(final_preds == 1)[0]
    if len(anomaly_indices) > 0:
        sample_indices = np.random.choice(anomaly_indices, min(5, len(anomaly_indices)), replace=False)
        print("  Berikut 5 sampel yang ditangkap:")
        for idx in sample_indices:
            row = df_processed.iloc[idx]
            true_label = "BENIGN (Salah Tangkap)" if y_true[idx] == 0 else "ANOMALY (Tangkapan Benar)"
            print(f"  - FileID: {row.get('file_id')} | Status Asli: {true_label}")
            print(f"    * Entropy: {row.get('global_entropy'):.3f} | NP_Ratio: {row.get('non_printable_ratio'):.3f} | Byte_Mean: {row.get('byte_mean'):.3f}")
    else:
        print("  [INFO] Tidak ada file yang ditandai sebagai Anomaly.")

    # 7 & 8. Gap CV vs Real & Konsistensi Skor Anomali
    print("\n[8] RENTANG SKOR ANOMALI")
    print(f"  Min Score : {raw_scores.min():.4f}")
    print(f"  Max Score : {raw_scores.max():.4f}")
    print(f"  Mean Score: {raw_scores.mean():.4f}")
    
    print("\n" + "="*60)
    print("🎯 KESIMPULAN AUDIT AI ANALIS:")
    if mod_meta.get('feature_count') != X.shape[1]:
        print("  ❌ MODEL RUSAK / BERBEDA VERSI. Feature training != Feature inference.")
        print("  💡 Tindakan: Wajib jalankan 'run_pipeline.py --action train' SEKARANG JUGA.")
    elif fp > 5:
        print("  ❌ False Positive masih tinggi.")
        print("  💡 Tindakan: Evaluasi threshold atau turunkan Contamination Ratio.")
    elif tf := meta.get('percentile') != PERCENTILE_VALUE if meta else True:
         print("  ⚠️ Model sudah stabil dari segi arsitektur baru, TAPI belum di-train ulang setelah ada perubahan Pct=56 atau fitur baru.")
         print("  💡 Tindakan: Wajib Retrain (python tools/run_pipeline.py) karena Config dan Data berubah.")
    else:
        print("  ✅ MODEL SEHAT WAL AFIAT!")
        print("  💡 Tindakan: Siap masuk jalur Production.")
    print("=" * 60)

if __name__ == "__main__":
    main()
