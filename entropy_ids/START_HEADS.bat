@echo off
color 0A
title H.E.A.D.S. AI Launcher

echo ===============================================================
echo     🛡️ MEMULAI H.E.A.D.S (Hybrid Entropy Anomaly System) 🛡️
echo ===============================================================
echo.
echo [*] Memastikan lokasi folder sudah benar...
cd /d "%~dp0"

echo [*] Menyalakan Mesin Pindai (Auto Scanner) di jendela terpisah...
:: Perintah 'start cmd /k' akan membuka jendela hitam baru khusus untuk CCTV
start "Mesin Scanner H.E.A.D.S" cmd /k "..\.venv\Scripts\python.exe tools\auto_scanner.py"

echo [*] Menyalakan Dashboard Visual (Streamlit Web UI)...
:: Perintah ini akan langsung membuka browser Anda
"..\.venv\Scripts\python.exe" -m streamlit run gui\dashboard.py

pause
