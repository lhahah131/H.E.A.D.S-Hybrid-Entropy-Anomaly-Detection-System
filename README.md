# 🛡️ H.E.A.D.S — Hybrid Entropy Anomaly Detection System (v1.0)

> **Tugas Evaluasi / Remidi | Evaluation / Remedial Assignment**
> 
> - **Name:** Adi Suryadi
> - **Semester:** 4
> - **Year:** 2026

> 🇮🇩 📖 **[Klik di sini untuk membaca Laporan Remidi (Versi Indonesia)](entropy_ids/docs/Laporan_Remidi_HEADS.md)**  
> 🇬🇧 📖 **[Click here to read the Remedial Report (English Version)](entropy_ids/docs/Remedial_Report_HEADS_EN.md)**

---

## 📌 Ringkasan Eksekutif | Executive Summary

**🇮🇩 (ID)**
Sistem H.E.A.D.S (v1.0) telah resmi dibekukan dan siap dioperasikan di lingkungan *Production*. Sistem ini mengombinasikan algoritma Isolation Forest dengan Benign Confirmation Layer khusus untuk mengenali anomali (*malware payloads*, terenkripsi) berdasarkan analisis fitur berbasis entropi dengan *false positive* yang mendekati nol.

Model telah diaudit dan menunjukkan Generalisasi yang andal di atas distribusi data riil tanpa mengalami over-sensitivitas (Alarm Rate < 30%).

**🇬🇧 (EN)**
The H.E.A.D.S (v1.0) system has been officially frozen and is ready for production deployment. It combines the Isolation Forest algorithm with a Benign Confirmation Layer to detect anomalies (*malware payloads*, encrypted files) using entropy-based feature analysis with near-zero false positives.

The model has been audited and demonstrates reliable generalization on real data distribution without over-sensitivity (Alarm Rate < 30%).

---

## 🏗️ Arsitektur Model Utama | Core Model Architecture

**🇮🇩 (ID)**
- **Algoritma Dasar:** `sklearn.ensemble.IsolationForest`
- **Jumlah Pohon (N_Estimators):** 300
- **Contamination Ratio:** 0.18
- **Persentil Threshold:** 56 (Locked/Persisted)

Sistem menggunakan mekanisme *Locked Threshold*, di mana threshold hanya dihitung saat fase pelatihan dan tidak dihitung ulang saat produksi.

**🇬🇧 (EN)**
- **Base Algorithm:** `sklearn.ensemble.IsolationForest`
- **Number of Trees (N_Estimators):** 300
- **Contamination Ratio:** 0.18
- **Percentile Threshold:** 56 (Locked/Persisted)

The system applies a *Locked Threshold* mechanism, meaning the anomaly boundary is calculated only during training and reused during production without recalculation.

---

## 🔄 Alur Kerja Sistem | Workflow

**🇮🇩 (ID)**
1. Input Data
2. Feature Extraction (11 metrik)
3. Isolation Forest Scoring
4. Threshold Evaluation
5. Benign Confirmation Layer
6. Verdict Output: `BENIGN` atau `ANOMALY`

**🇬🇧 (EN)**
1. Data Ingestion
2. Feature Extraction (11 metrics)
3. Isolation Forest Scoring
4. Threshold Evaluation
5. Benign Confirmation Layer
6. Final Verdict: `BENIGN` or `ANOMALY`

---

## 🛡️ Benign Confirmation Layer (Strict Mode)

**🇮🇩 (ID)**
File dinyatakan **Aman** jika:
- `ascii_ratio > 0.85`
- `non_printable_ratio < 0.05`
- `global_entropy < 4.8`

Anomaly dikunci **positif** jika:
- `global_entropy > 4.75`
- `non_printable_ratio > 0.015`

*Layer ini menurunkan False Positive secara signifikan.*

**🇬🇧 (EN)**
A file is considered **Safe** if:
- `ascii_ratio > 0.85`
- `non_printable_ratio < 0.05`
- `global_entropy < 4.8`

An anomaly is confirmed **only if**:
- `global_entropy > 4.75`
- `non_printable_ratio > 0.015`

*This layer significantly reduces false positives.*

---

## 💻 Panduan Menjalankan Sistem | Usage Commands

**🇮🇩 (ID)**
Arahkan direktori terminal ke dalam folder `entropy_ids` terlebih dahulu, lalu gunakan perintah berikut:

1. **Melatih & Menguji Model (Training & CV) 🧪**
   *Mengeksekusi pipeline pelatihan penuh dengan parameter model 1.0:*
   ```bash
   python app/main.py --action train --mode adaptive
   ```

2. **Deteksi/Inference Real-Time (Production) 🚀**
   *Menganalisis file tak dikenal murni menggunakan threshold dari model produksi:*
   ```bash
   python app/main.py --action inference
   ```

3. **Audit Kualitas Produksi & Threshold 🔍**
   *Membaca statistik model dan mengecek kebocoran threshold dinamis:*
   ```bash
   python tools/audit_model.py
   ```

4. **Monitoring Dashboard Langsung 📊**
   *Membuka live-dashboard untuk memantau trafik anomali secara visual:*
   ```bash
   python tools/dashboard_monitor.py
   ```

**🇬🇧 (EN)**
Navigate your terminal into the `entropy_ids` folder first, then execute the following commands:

1. **Train & Evaluate Model (Training & CV) 🧪**
   *Executes the full training pipeline using the 1.0 model parameters:*
   ```bash
   python app/main.py --action train --mode adaptive
   ```

2. **Real-Time Inference (Production) 🚀**
   *Analyzes fresh unknown files using purely the frozen production threshold:*
   ```bash
   python app/main.py --action inference
   ```

3. **Production Quality & Threshold Audit 🔍**
   *Reads model statistics and audits for dynamic threshold leakage:*
   ```bash
   python tools/audit_model.py
   ```

4. **Live Monitoring Dashboard 📊**
   *Deploys the live-terminal dashboard to visually monitor anomaly traffic:*
   ```bash
   python tools/dashboard_monitor.py
   ```