import streamlit as st
import pandas as pd
import os
import time
import subprocess
import sys
from datetime import datetime

# Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="H.E.A.D.S Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan (Dark Mode Premium)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .status-aman {
        color: #00ff00;
        font-weight: bold;
    }
    .status-blokir {
        color: #ff3333;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Lokasi File Log Sejarah Scanner (Satu tingkat ke atas karena script ini ada di folder /gui/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILEPATH = os.path.join(ROOT_DIR, "logs", "scan_history.log")

def load_log_data():
    """Fungsi untuk membaca file log dan mengubahnya menjadi tabel DataFrame"""
    if not os.path.exists(LOG_FILEPATH):
        return pd.DataFrame(columns=["Waktu", "File", "Status", "Entropi", "Non-Printable (%)"])
        
    data = []
    try:
        with open(LOG_FILEPATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip(): continue
                # Parsing baris log: [2026-02-28 23:10:05] FILE: nama.txt | STATUS: DITERIMA (AMAN) | ENTROPI: 3.511 | NON-PRINTABLE: 4.1%
                parts = line.strip().split(" | ")
                if len(parts) >= 4:
                    time_str = parts[0].split("]")[0].replace("[", "").strip()
                    file_name = parts[0].split("FILE: ")[1].strip() if "FILE: " in parts[0] else "Unknown"
                    status = parts[1].replace("STATUS: ", "").strip()
                    entropy = float(parts[2].replace("ENTROPI: ", "").strip())
                    np_ratio = float(parts[3].replace("NON-PRINTABLE: ", "").replace("%", "").strip())
                    
                    data.append({
                        "Waktu": time_str,
                        "File": file_name,
                        "Status": status,
                        "Entropi": entropy,
                        "Non-Printable (%)": np_ratio
                    })
    except Exception as e:
        st.error(f"Gagal membaca log: {e}")
        
    df = pd.DataFrame(data)
    # Urutkan dari yang terbaru
    if not df.empty:
        df = df.sort_values(by="Waktu", ascending=False).reset_index(drop=True)
    return df

# Main Header
st.markdown('<p class="big-font">🛡️ H.E.A.D.S. Real-Time Security Center</p>', unsafe_allow_html=True)
st.markdown("*Hybrid Entropy Anomaly Detection System - Version 1.0*")
st.divider()

# --- SIDEBAR: PANEL KENDALI SCANNER ---
st.sidebar.markdown('<h2>⚙️ Panel Kendali AI</h2>', unsafe_allow_html=True)
st.sidebar.markdown("Nyalakan / Matikan mesin Antivirus dari sini:")

if "scanner_process" not in st.session_state:
    st.session_state.scanner_process = None

if st.session_state.scanner_process is None:
    if st.sidebar.button("▶️ HIDUPKAN SCANNER", type="primary"):
        scanner_script = os.path.join(ROOT_DIR, "tools", "auto_scanner.py")
        venv_python = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")
        
        # Cek apakah dijalankan dalam virtual env
        if not os.path.exists(venv_python):
            venv_python = sys.executable
            
        # Panggil secara asinkron tanpa jendela hitam tambahan (Ghaib/Siluman)
        proc = subprocess.Popen([venv_python, scanner_script], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        st.session_state.scanner_process = proc
        st.sidebar.success("✅ Detektor aktif menjaga PC!")
        time.sleep(0.5)
        st.rerun()
else:
    st.sidebar.success("✅ Detektor Sedang Berjaga!")
    if st.sidebar.button("⏹️ MATIKAN SCANNER", type="secondary"):
        st.session_state.scanner_process.kill()
        st.session_state.scanner_process = None
        st.sidebar.warning("🛑 Detektor dimatikan.")
        time.sleep(0.5)
        st.rerun()

st.sidebar.divider()
st.sidebar.info(f"💡 Lokasi Penjagaan:\\n...\\\\data\\\\sandbox")

# Tombol Refresh Manual atau Auto-Refresh (Ditaruh di bawah)
col_auto, col_space = st.columns([2, 8])
with col_auto:
    auto_refresh = st.checkbox("Auto-Refresh (Setiap 3 Detik)", value=True)

# --- Mengambil Data Real-Time ---
df_logs = load_log_data()

# --- BAGIAN ATAS: KARTU METRIK STATISTIK ---
col1, col2, col3, col4 = st.columns(4)

total_scanned = len(df_logs)
total_blocked = len(df_logs[df_logs["Status"] == "DIBLOKIR (MALWARE)"]) if not df_logs.empty else 0
total_safe = len(df_logs[df_logs["Status"] == "DITERIMA (AMAN)"]) if not df_logs.empty else 0

if total_scanned > 0:
    block_rate = (total_blocked / total_scanned) * 100
else:
    block_rate = 0.0

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="Total File Dipindai", value=total_scanned)
    st.markdown('</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="⚠️ Ancaman Diblokir", value=total_blocked, delta=f"{block_rate:.1f}% Rasio", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)
    
with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label="✅ File Aman", value=total_safe)
    st.markdown('</div>', unsafe_allow_html=True)
    
with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_entropy = df_logs["Entropi"].mean() if not df_logs.empty else 0.0
    st.metric(label="Rata-rata Entropi", value=f"{avg_entropy:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("") # Spasi
st.divider()

# --- BAGIAN TENGAH: GRAFIK VISUALISASI ---
if not df_logs.empty:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Aktivitas Entropi File Terakhir")
        # Menampilkan grafik garis (Line Chart) tren entropi
        chart_data = df_logs.head(20).copy() # Ambil 20 file terakhir
        chart_data = chart_data.sort_values("Waktu", ascending=True) # Urutkan kronologis untuk grafik
        st.line_chart(chart_data, x="File", y="Entropi", height=300, color="#ff4b4b")
        
    with col_chart2:
        st.subheader("📊 Distribusi Non-Printable Bytes")
        # Grafik batang untuk melihat kekacauan byte tak kasat mata
        st.bar_chart(chart_data, x="File", y="Non-Printable (%)", height=300, color="#00d2ff")
else:
    st.info("Menunggu data masuk... Silakan hidupkan auto_scanner.py dan jatuhkan file ke dalam folder Sandbox.")

st.divider()

# --- BAGIAN BAWAH: TABEL LOG RIWAYAT (LIVE ENTRY) ---
st.subheader("📋 Riwayat Pemindaian (Live Log)")

# Fungsi pewarnaan tabel
def highlight_status(val):
    if "DIBLOKIR" in str(val):
        return 'color: #ff4b4b; font-weight: bold'
    elif "AMAN" in str(val):
        return 'color: #00ff00; font-weight: bold'
    return ''

if not df_logs.empty:
    # Tampilkan DataFrame dengan st.dataframe dan style map
    styled_df = df_logs.style.map(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, height=400)
else:
    st.write("Belum ada file yang dipindai hari ini.")

# Tombol untuk membersihkan log
if st.button("🗑️ Bersihkan Riwayat Log"):
    if os.path.exists(LOG_FILEPATH):
        with open(LOG_FILEPATH, "w") as f:
            f.write("")
        st.toast("Riwayat Log Berhasil Dikosongkan!", icon="✅")
        time.sleep(1)
        st.rerun()

# EKSEKUSI AUTO-REFRESH (HARUS DITARUH PALING BAWAH!)
# Jika ditaruh di atas, script tidak akan pernah mengeksekusi (merender) tampilan UI ke bawah
if auto_refresh:
    time.sleep(3)
    st.rerun()
