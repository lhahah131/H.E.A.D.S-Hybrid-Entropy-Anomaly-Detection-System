# 🛡️ H.E.A.D.S — Hybrid Entropy Anomaly Detection System (v1.0)

> **Tugas Evaluasi / Remidi**
> - **Nama:** Adi Suryadi
> - **Semester:** 4
> - **Tahun:** 2026
>
> 🔗 **[Repository GitHub](https://github.com/lhahah131/H.E.A.D.S-Hybrid-Entropy-Anomaly-Detection-System-v1.0-.git)**
>
> 📖 **[Klik di sini untuk membaca Laporan Remidi Selengkapnya](entropy_ids/docs/Laporan_Remidi_HEADS.md)**

---

## 📌 Ringkasan Eksekutif
Sistem **H.E.A.D.S (v1.0)** telah resmi dibekukan dan siap dioperasikan di lingkungan *Production*. Sistem ini mengombinasikan algoritma **Isolation Forest** dengan **Benign Confirmation Layer** khusus untuk mengenali anomali *(malware payloads, terenkripsi)* berdasarkan analisis fitur berbasis entropi dengan false positive yang mendekati nol.

Model telah diaudit dan menunjukkan Generalisasi yang andal di atas distribusi data riil tanpa mengalami over-sensitivitas *(Alarm Rate < 30%)*.

---

## 🏗️ Arsitektur Model Utama
- **Algoritma Dasar:** `sklearn.ensemble.IsolationForest`
- **Jumlah Pohon (N_Estimators):** 300
- **Contamination Ratio:** 0.18 (Tingkat agresivitas telah dikalibrasi realistis dari 0.35)
- **Persentil Threshold:** 56 (Locked/Persisted)

Sistem menggunakan **Locked Threshold Mechanism** pada tahapan komputasinya. Artinya, kalkulasi batas pemutus anomali (Threshold) hanya berlaku dan dihitung pada fase **Train**. Pada fase **Inference / Production**, model murni menarik angka hasil panen pelatihan *(Zero Dynamic Recalculation)* demi menjaga stabilitas pelacakan drift.

---

## 🔄 Alur Kerja Sistem (Workflow)
Secara garis besar, pendeteksian berjalan otomatis melalui tahap berikut:
1. **Input Data:** Menerima *file* statis yang disandikan ke dalam representasi log.
2. **Feature Extraction:** Menghitung susunan 11 metrik fitur (Keacakan Entropi & Rasio Byte).
3. **Isolation Forest Scoring:** Mengukur angka "Keanehan" *file* dari struktur kepadatan algoritma pohon.
4. **Threshold Evaluation:** Mengadu skor keanehan dengan batas mutlak (*Persisted Threshold*).
5. **Benign Confirmation Layer:** Memeriksa kembali klaim "Anomali" untuk mengeksekusi pencegahan *False Positive*.
6. **Verdict / Output:** Melemparkan status akhir keputusan: `BENIGN` (Aman) atau `ANOMALY` (Ancaman).

---

## 🧬 Feature Engineering (Top 11 Features)
Sistem V1.0 menggunakan *Feature Density Optimization*. Atribut lemah dan tidak informatif telah dicukur bersih untuk menghindari kebingungan model (Curse of Dimensionality). Fitur kunci yang tersisa:
1. `global_entropy`
2. `block_mean_entropy`
3. `block_std_entropy`
4. `non_printable_ratio`
5. `ascii_ratio`
6. `byte_mean`
7. `byte_std`
8. `byte_skewness`
9. **`entropy_x_nonprint`** *(Kombinasi Baru - Kuat untuk sandi terselubung)*
10. **`entropy_div_ascii`** *(Kombinasi Baru)*
11. **`bytestd_div_bytemean`** *(Kombinasi Baru)*

---

## 🛡️ Benign Confirmation Layer (Strict Mode)
Fitur unggulan V1.0 adalah validasi ganda dari hasil klaim anomali. Setiap deteksi (label=1) dari hutan algoritma wajib menjalani validasi logika keras:
- File dianulir dan dikembalikan ke status **Aman (0)** JIKA:
  1. `ascii_ratio` > 0.85 DAN
  2. `non_printable_ratio` < 0.05 DAN
  3. `global_entropy` < 4.8
- Anomaly dikunci positif HANYA JIKA:
  `global_entropy` > 4.75 DAN `non_printable_ratio` > 0.015

Layer ini menyumbang penurunan luar biasa terhadap angka *False Positive* (Aman dituduh Malware). 

---

## 📊 Metrik Evaluasi Produksi Akhir
Diuji di atas 100 Real-Simulation Data Master Database:
- **Real F1 Score      :** 0.7636
- **Precision          :** 0.8400
- **Recall             :** 0.7000
- **ROC AUC            :** 0.8467
- **False Positives    :** Hanya 4 Alarm Palsu (Turun drastis dari 31 Kasus di Beta V4)
- **False Negatives    :** 9  
- **Flagged Alarm Rate :** 25.0%

---

## 💻 Panduan Menjalankan Sistem

1. **Untuk Melatih & Menguji Model (Pipeline Penuh):**
   ```bash
   python entropy_ids/tools/run_pipeline.py
   ```

2. **Untuk Menjalankan Audit Kualitas (Health Check):**
   ```bash
   python entropy_ids/tools/audit_model.py
   ```

3. **Untuk Membuka Monitoring Dashboard (Live Terminal):**
   ```bash
   python entropy_ids/tools/dashboard_monitor.py
   ```

*(Log Eksekusi & Bukti Audit tersimpan otomatis dalam histori log di direktori `entropy_ids/logs/`)*
