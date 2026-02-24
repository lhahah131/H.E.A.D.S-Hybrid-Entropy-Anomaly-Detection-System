# 📄 Laporan Remidi
**H.E.A.D.S — Hybrid Entropy Anomaly Detection System (v1.0)**

---

## Identitas
- **Nama:** Adi Suryadi
- **Semester:** 4
- **Tahun:** 2026

---

## 1. Ringkasan Proyek
Pada remidi/evaluasi ini, saya mengembangkan sebuah sistem deteksi anomali berbasis *Machine Learning* untuk menganalisis *file* statis dan mengidentifikasi potensi ancaman *malware* berdasarkan karakteristik statistik tingkat rendah *(byte-level)* seperti *entropy* dan distribusi siluman (*byte*).

Sistem ini berevolusi menggunakan algoritma **Isolation Forest** dengan pendekatan **Hybrid Anomaly Detection**. Arsitektur akhirnya dibangun agar sistem deteksi ini stabil, konsisten, serta siap digunakan secara langsung dalam skenario pemantauan lapangan (monitoring).

## 2. Konsep dan Arsitektur yang Digunakan
Sistem ini menggunakan beberapa sub-komponen utama:
- **Feature Extraction:** Transformasi berbasis tingkat keacakan (entropy) dan statistik byte, mencakup 11 fitur utama dengan optimasi kepadatan data.
- **Model Isolation Forest:** Algoritma pendeteksi dengan parameter 300 *estimators*.
- **Locked Persisted Threshold:** Penguncian mutlak pada nilai batas deteksi (threshold tidak dihitung ulang secara dinamis saat tahap *inference*).
- **Metrik Evaluasi Realistis:** Pengamatan mendalam menggunakan *Precision*, *Recall*, dan *F1 Score*.
- **Production Audit Monitoring:** Uji coba peredaman *False Positive* dan audit stabilitas model sehari-hari.

Pendekatan ini difokuskan murni pada deteksi perilaku anomali (*Anomaly Detection*), bukan sekadar pengenalan pola *signature malware* tertentu peninggalan lawas.

### Alur Kerja Sistem (Workflow)
Untuk memberikan gambaran operasional, berikut adalah rincian spesifik bagaimana data bergerak di dalam sistem **H.E.A.D.S**:
1. **Fase Ingesti (Data Ingestion)**
   Sistem membaca baris data ekstraksi dari *file* mentah yang sedang diperiksa targetnya.
2. **Fase Ekstraksi (Feature Engineering)**
   Mengalkulasi metrik *Global Entropy*, *Block Entropy*, serta rasio karakter terselubung (*Non-printable Ratio*). Lalu membangun kombinasi khusus untuk memburu sandi (seperti: `entropy_x_nonprint`).
3. **Fase Isolasi (Isolation Forest Engine)**
   11 Fitur matang tersebut dikirim ke dalam hutan pohon keputusan `IsolationForest`. Model kemudian memuntahkan skor "Keterasingan" (*Raw Anomaly Score*).
4. **Fase Pengukuran Bawah Sadar (Locked Thresholding)**
   Skor mentah dibandingkan dengan angka ambang mutlak dari metadata *Training* (*Persisted Threshold*). Jika skor menunjukkan sinyal negatif ekstrem di bawah garis tapal batas, asumsikan ancaman awal bertatus *Malware*.
5. **Fase Konfirmasi Medis (Benign Confirmation Layer)**
   Berkas yang dicurigai sebagai *Malware* dikunci jalannya oleh filter **BCL**. Jika sistem membaca porsi teks standar (ASCII > 85%) dan tingkat keacakan wajar (Entropy < 4.8), maka *"tuduhan*" tersebut batal dan status ditarik kembali ke garis Aman (*Benign/0*).
6. **Fase Putusan Final (Final Verdict)**
   Status diterbitkan (Aman / Ancaman). Rekam metrik *False Positive* atau hasil tangkapan dieksekusi masuk ke sistem riwayat operasional (Logs).

#### 🔀 Flowchart Arsitektur H.E.A.D.S
Diagram alur ini merepresentasikan bagaimana lalu lintas *file* bergerak melintasi sistem dari hulu (ekstraksi data) ke hilir (pengambilan keputusan rilis ancaman). (Dapat dilihat secara visual jika *markdown editor* mendukung ekstensi `mermaid`):

+--------------------------------------------------+
|           INPUT FILE STATIS & LOG               |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|     FEATURE EXTRACTION & ENGINEERING            |
|               (11 METRIK)                       |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|           ISOLATION FOREST MODEL                |
|     (Perhitungan Skor Keterasingan)             |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|              THRESHOLD EVALUATION               |
+--------------------------------------------------+
            |                          |
            |                          |
            v                          v
   Skor > Threshold             Skor <= Threshold
   (Gejala Normal)              (Gejala Anomali)
            |                          |
            v                          v
+--------------------+        +-----------------------------+
| STATUS: BENIGN     |        | BENIGN CONFIRMATION LAYER   |
| / AMAN             |        +-----------------------------+
+--------------------+                   |
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | ASCII > 85% & Entropy < 4.8 |
            |                   | (Teks Wajar)                 |
            |                   +-----------------------------+
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | STATUS: DIBATALKAN / AMAN   |
            |                   +-----------------------------+
            |                              |
            |                              |
            |                   +-----------------------------+
            |                   | Non-Printable > 0.015       |
            |                   | & Entropy > 4.75            |
            |                   | (File Asimetris)            |
            |                   +-----------------------------+
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | STATUS: MALWARE / ANCAMAN   |
            |                   +-----------------------------+
            |                              |
            +--------------+---------------+
                           |
                           v
                +-------------------------+
                | SISTEM LOG & REPORTING |
                +-------------------------+

## 3. Proses Metodologi Pengembangan
Pengembangan dilakukan secara sistematis melalui beberapa tahapan *MLOps*:
1. **Analisis Ekstraksi:** Eksplorasi distribusi data dan desain fitur.
2. **Pelatihan Model:** Konstruksi kerangka awal dan pengujian lab dengan metode *5-Fold Cross-Validation*.
3. **Pengujian Data Riil:** Uji pada data nyata untuk meraba jumlah *Alarm Palsu (False Positive)* dan menyimak seberapa besar deviasi/gap yang terlahir.
4. **Stabilisasi Model (Production Phase):** Fokus utama ditarik pada penjagaan keseimbangan *Precision* vs *Recall* serta meminimalkan alarm yang tidak perlu demi penggunaan jangka panjang.

## 4. Hasil Evaluasi Model
Model akhir dari arsitektur (*Version 1.0 Stable*) memperlihatkan metrik operasional terjamin:
- **Precision Tinggi dan Stabil:** Menggambarkan minimnya tingkat *False Positive* (sangat aman dari alarm palsu).
- **Recall Stabil:** Menahan pergerakan anomali jahat agar tidak lolos penjagaan.
- **Gap Evaluasi vs Real: Kecil.** Algoritme terbukti menguasai generalisasi pola lapangan tanpa gejala *'Overfitting'*.
- **Zero Data Drift:** Tidak ditemukan adanya penyimpangan *(drift)* atau distorsi data yang signifikan.
- **Konsistensi Lintas Lingkungan:** Model mengunci nilai *threshold* yang sama persis antara *training* eksekutif dengan *inference* operasional.

Sistem dikategorikan sebagai sistem monitoring mandiri yang stabil, awet, dan siap untuk dilepas atau dipakai sebagai fondasi pengembangan lanjutan.

## 5. Penggunaan AI dalam Pengembangan
Dalam proses iterasi penyusunan piranti lunak ini, saya mendayagunakan instrumen **AI (ChatGPT)** sebagai alat bantu (*Tooling Support*) dalam ranah spesifikasi operasi teknis, seperti:
- *Brainstorming* ide struktural / taktik arsitektur sistem.
- Bantuan *Refactoring Code* demi kebersihan kode (*Clean Code Architecture*).
- Penyusunan kerangka draft dokumentasi secara estetik.
- Validasi introspeksi dari logika fungsi kalkulasi metrik teknis.

Meski demikian, **seluruh desain inti, metodologi pengujian, interpretasi analisis hasil evaluasi (*Trade-off*), hingga segala keputusan akhir strategis teknikal dilakukan dan diverifikasi secara mandiri**. AI berkedudukan sebatas medium yang mendongkrak produktivitas operasional (*Typing Assistant*), sedangkan pengambilan keputusan klinis arsitektur tetap dikuasai sepenuhnya oleh naluri pengembang secara sadar.

## 6. Kesimpulan
Sistem ini sukses direalisasikan sebagai kerangka dasar inspeksi Anomali Hybrid berbasis analisis statis yang stabil, tanpa kerentanan kalkulasi *runtime*, dan sangat matang untuk dibangun versi lanjutannya nanti. Proyek ini mendemonstrasikan bukti pemahaman empiris ihwal MLOps, bertindak mandiri sebagai lapisan keamanan dini tanpa diwajibkan menjadi Antivirus penuh.

---

## 7. Cara Penggunaan Sistem H.E.A.D.S
Berikut adalah instruksi praktis untuk menjalankan model V1.0 di terminal/Command Prompt:

### A. Menjalankan Pipeline Lengkap (Train & Test)
Gunakan perintah ini untuk melatih ulang model menggunakan data terbaru, melihat skor validasi silang (Cross-Validation), dan langsung mengujinya (Inference):
```bash
python tools/run_pipeline.py
```

### B. Mengaudit Performa & Kesehatan Model
Untuk membaca rincian mendalam mengenai tingkat Alarm Palsu (False Positive), evaluasi Threshold yang sedang aktif, serta inspeksi sampel *"malware"* yang tertangkap:
```bash
python tools/audit_model.py
```

### C. Membuka Dashboard Pemantauan Livelog (Monitoring)
Sebagai seorang SysAdmin, Anda bisa memantau kesehatan skor perburuan data secara aktual (*real-time update*). Dashboard ini tidak akan memakan resource berat:
```bash
python tools/dashboard_monitor.py
```
*(Tekan `CTRL+C` di terminal untuk keluar dari layar dashboard monitoring).*
