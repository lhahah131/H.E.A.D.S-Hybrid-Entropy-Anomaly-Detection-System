# 🛡️ H.E.A.D.S — Hybrid Entropy Anomaly Detection System

<div align="center">

![Version](https://img.shields.io/badge/Versi-1.5-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Algorithm](https://img.shields.io/badge/AI-Isolation%20Forest%20%2B%20HWCL-purple)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

> **Tugas Evaluasi / Remidi Akhir**
> - **Nama:** Adi Suryadi
> - **Semester:** 4 — Tahun: 2026

📖 **[Klik di sini → Baca Laporan Resmi Remidi H.E.A.D.S v1.5 (Bahasa Indonesia)](docs/Laporan_Remidi_HEADS_v1.5.md)**

</div>

---

## 📌 Tentang Sistem Ini
**H.E.A.D.S.** adalah sebuah sistem *Next-Generation Antivirus (NGAV)* berbasis *Machine Learning* yang mampu mendeteksi *malware* secara **Real-Time** tanpa memerlukan database tanda tangan virus (*Signature-Free Detection*).

Sistem ini tidak hanya mendeteksi virus dari nilai Entropi-nya saja, melainkan juga membedah **struktur dalam aplikasi** (seperti *PE Header* dan *API calls berbahaya*) untuk mengenali segala macam teknik persembunyian *Malware* modern termasuk: *Packing*, *Encryption*, *Obfuscation*, dan *PowerShell Dropper*.

---

## 🏆 Hasil Evaluasi v1.5
| Metrik | v1.0 (Lama) | v1.5 (Terbaru) |
| :--- | :---: | :---: |
| **F1-Score** | 0.76 | **0.8889 ✅** |
| **ROC AUC** | 0.85 | **0.9471 ✅** |
| **Jumlah Fitur** | 11 | **16 ✅** |
| **PE Forensics** | ❌ | **✅** |
| **Dashboard GUI** | ❌ | **✅** |

---

## ⚡ Cara Menjalankan Sistem (Quick Start)

### 1. Instalasi
```bash
# Buat dan aktifkan Virtual Environment
python -m venv .venv
.venv\Scripts\activate

# Instal semua dependensi
pip install pandas numpy scikit-learn watchdog streamlit pefile
```

### 2. Latih Model AI
```bash
python tools/run_pipeline.py
```

### 3. Hidupkan Antivirus + Dashboard (1-Klik)
Cukup klik dua kali file:
```
START_HEADS.bat
```

### 4. Uji Coba Simulasi Malware
```bash
python tools/test_scanner.py
```

---

## 🗂️ Struktur Proyek
```
entropy_ids/
├── app/
│   └── core/          → Jantung AI (Feature Engine, HWCL, Isolation Forest)
├── data/
│   ├── raw/           → Dataset Master CSV
│   └── sandbox/       → Zona Karantina File (tempat Scanner mengintai)
├── docs/
│   └── Laporan_Remidi_HEADS_v1.5.md  → 📄 Laporan Resmi Proyek
├── gui/
│   └── dashboard.py   → Pusat Komando Visual (Streamlit)
├── models/            → Memori AI (.pkl)
├── tools/             → Alat Tempur (Scanner, Pipeline, Tester)
├── logs/              → Arsip Audit Historis
└── START_HEADS.bat    → 🚀 Tombol 1-Klik
```
