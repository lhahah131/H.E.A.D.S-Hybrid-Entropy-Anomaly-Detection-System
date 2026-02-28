import os
import time
import random
import string
import base64

def generate_files():
    sandbox_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sandbox")
    os.makedirs(sandbox_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 🦠 MEMULAI SIMULASI SERANGAN MULTI-MALWARE 🦠")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. FILE NORMAL (BENIGN)
    # ---------------------------------------------------------
    benign_path = os.path.join(sandbox_dir, "test_1_dokumen_aman.txt")
    print(f"\n[1] Menjatuhkan File AMAN: {os.path.basename(benign_path)}")
    with open(benign_path, "w") as f:
        f.write("Halo, ini laporan keuangan bulan ini.\n")
        f.write("Semua data terlihat normal dan wajar.\n" * 10)
    time.sleep(4) # Tunggu CCTV memproses
    
    # ---------------------------------------------------------
    # 2. RANSOMWARE ENCRYPTED DATA (High Entropy, High Non-Printable)
    # Tipe serangan: File pengguna yang telah dienkripsi ransomware modern (LockBit, dll)
    # ---------------------------------------------------------
    ransomware_path = os.path.join(sandbox_dir, "test_2_ransomware_encrypted.bin")
    print(f"\n[2] Menjatuhkan RANSOMWARE (Data Terenkripsi): {os.path.basename(ransomware_path)}")
    # Murni byte acak = Entropi nyaris 8.0 maksimal.
    random_bytes = bytearray(random.getrandbits(8) for _ in range(30000))
    with open(ransomware_path, "wb") as f:
        f.write(random_bytes)
    time.sleep(4)
    
    # ---------------------------------------------------------
    # 3. BASE64 POWERSHELL DROPPER (Med-High Entropy, High ASCII)
    # Tipe serangan: Script teks yang tidak punya karakter aneh sama sekali (100% ASCII Printables), 
    # namun isinya menyimpan virus tersembunyi berwujud base64 raksasa.
    # ---------------------------------------------------------
    b64_dropper_path = os.path.join(sandbox_dir, "test_3_base64_dropper.ps1")
    print(f"\n[3] Menjatuhkan SCRIPT DROPPER (Base64 PowerShell): {os.path.basename(b64_dropper_path)}")
    # Payload aslinya acak, tapi di-encode ke Base64 (Text biasa)
    hidden_payload = bytes([random.randint(0, 255) for _ in range(40000)])
    b64_payload = base64.b64encode(hidden_payload).decode('utf-8')
    with open(b64_dropper_path, "w") as f:
        f.write("$encodedData = '" + b64_payload + "';\n")
        f.write("iex ([System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($encodedData)))")
    time.sleep(4)
    
    # ---------------------------------------------------------
    # 4. PACKED MALWARE EXECUTABLE (Header Normal + Body Acak)
    # Tipe serangan: File .exe yang di-pack/dikompres agar tidak terdeteksi Antivirus tradisional (misal: UPX Packed Trojan)
    # ---------------------------------------------------------
    packed_exe_path = os.path.join(sandbox_dir, "test_4_packed_trojan.exe")
    print(f"\n[4] Menjatuhkan PACKED TROJAN (.EXE Terkompresi): {os.path.basename(packed_exe_path)}")
    
    # Header Windows PE/EXE Asli yang sangat normal (Low Entropy)
    mz_header = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe8\x00\x00\x00\x0e\x1f\xba\x0e\x00\xb4\x09\xcd\x21\xb8\x01\x4c\xcd\x21\x54\x68\x69\x73\x20\x70\x72\x6f\x67\x72\x61\x6d\x20\x63\x61\x6e\x6e\x6f\x74\x20\x62\x65\x20\x72\x75\x6e\x20\x69\x6e\x20\x44\x4f\x53\x20\x6d\x6f\x64\x65\x2e\x0d\x0d\x0a\x24\x00\x00\x00\x00\x00\x00\x00"
    
    # Body yang sangat terenkripsi murni (High Entropy)
    packed_body = bytearray(random.getrandbits(8) for _ in range(45000))
    
    with open(packed_exe_path, "wb") as f:
        # Tulis secara perlahan seperti orang sedang mendownload Trojan dari internet 
        f.write(mz_header)
        f.flush()
        time.sleep(1)
        for i in range(0, len(packed_body), 15000):
            f.write(packed_body[i:i+15000])
            f.flush()
            time.sleep(1) # Sensimulasi download lambat, menguji "Patience Sensor" 15 Detik di auto_scanner
            
    print("\n" + "=" * 60)
    print(" [+] Simulasi Selesai! Silakan cek hasil tilang/keputusan di Terminal Auto Scanner!")
    print("=" * 60)

if __name__ == "__main__":
    generate_files()
