# 🛡️ H.E.A.D.S — Hybrid Entropy Anomaly Detection System (v1.5)

<div align="center">

![Version](https://img.shields.io/badge/Versi-1.5-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Algorithm](https://img.shields.io/badge/AI-Isolation%20Forest%20%2B%20HWCL-purple)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

> **Tugas Evaluasi / Remidi Akhir**
> - **Nama:** Adi Suryadi
> - **Semester:** 4 — Tahun: 2026

🇮🇩 📖 **[Klik di sini → Baca Laporan Resmi Remidi H.E.A.D.S v1.5](entropy_ids/docs/Laporan_Remidi_HEADS_v1.5.md)**

</div>

---

## 📌 Ringkasan Eksekutif
Sistem **H.E.A.D.S. v1.5** adalah sebuah *Next-Generation Antivirus (NGAV)* berbasis *Machine Learning* yang mampu mendeteksi *malware* secara **Real-Time** tanpa database tanda tangan virus (*Signature-Free*). 

Berevolusi dari v1.0 yang hanya mengandalkan 11 Fitur Entropi, versi terbaru ini kini memiliki **16 Fitur Cerdas** termasuk analisis forensik *Portable Executable (PE Header)*, deteksi API berbahaya, dan analisis teks mencurigakan yang membuatnya tahan terhadap teknik persembunyian *Malware* termutakhir seperti *Packing*, *Encryption*, dan *PowerShell Dropper*.

---

## 🏆 Perbandingan Kinerja: v1.0 vs v1.5

| Metrik | v1.0 (Lama) | v1.5 (Sekarang) |
| :--- | :---: | :---: |
| **F1-Score** | 0.76 | **0.8889 ✅** |
| **ROC AUC** | 0.85 | **0.9471 ✅** |
| **Jumlah Fitur AI** | 11 | **16 ✅** |
| **PE Forensics (pefile)** | ❌ | **✅** |
| **Dashboard GUI Real-Time** | ❌ | **✅** |
| **1-Klik Launcher (.bat)** | ❌ | **✅** |

---

## 🔄 Alur Kerja Sistem
1. **Pengintaian:** `auto_scanner.py` berjalan diam-diam mengawasi folder `data/sandbox`.
2. **Ekstraksi Super:** Begitu file masuk, sistem membedah **16 Fitur** (Entropi + PE Header + String Forensik).
3. **Keputusan AI:** Model *Isolation Forest + HWCL* memproses fitur dalam <1 detik.
4. **Laporan Visual:** Vonis **✅ AMAN** atau **❌ DIBLOKIR** muncul langsung di Streamlit Dashboard.

---

## ⚡ Cara Menjalankan (Quick Start)

### 1. Instalasi
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn watchdog streamlit pefile
```

### 2. Latih Ulang Model AI (Opsional)
```bash
cd entropy_ids
python tools/run_pipeline.py
```

### 3. Hidupkan Antivirus + Dashboard (1-Klik!)
```
Klik 2x: entropy_ids/START_HEADS.bat
```

### 4. Uji Serangan Malware (Simulasi)
```bash
cd entropy_ids
python tools/test_scanner.py
```

---

## 🗂️ Struktur Proyek
```
entropy_ids/
├── app/core/     → Jantung AI (Feature Engine, HWCL, Isolation Forest)
├── data/sandbox/ → Zona Karantina File Real-Time
├── docs/         → 📄 Laporan Resmi Remidi v1.5
├── gui/          → Dashboard Streamlit (Web Monitor)
├── models/       → Memori AI (.pkl)
├── tools/        → Scanner, Pipeline, Tester, Upgrade
├── logs/         → Arsip Audit Historis
└── START_HEADS.bat → 🚀 Tombol 1-Klik Peluncur
```