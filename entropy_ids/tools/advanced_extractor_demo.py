import pefile
import os
import math

def extract_advanced_features(filepath):
    """
    Contoh Simulasi Ekstraksi Fitur Lanjut (PE Header & Forensik String) 
    untuk disuapkan ke dalam Model Machine Learning Klasik.
    """
    advanced_features = {
        "is_executable": 0,
        "num_sections": 0,
        "suspicious_api_count": 0,
        "has_high_entropy_section": 0,
        "suspicious_string_count": 0
    }
    
    # 1. FORENSIK STRING (Sangat RIngan, Berlaku untuk semua file)
    try:
        with open(filepath, "rb") as f:
            data = f.read()
            
        # Mencari string yang sering dipakai Hacker menyembunyikan URL/Perintah
        suspicious_keywords = [b"http://", b"https://", b"powershell", b"cmd.exe", b"WScript.Shell"]
        keyword_count = sum(data.count(kw) for kw in suspicious_keywords)
        advanced_features["suspicious_string_count"] = keyword_count
    except Exception:
        pass

    # 2. FORENSIK PE/HEADER (Hanya berlaku jika file adalah .exe/.dll)
    try:
        # pefile akan membongkar "paspor" file exe
        pe = pefile.PE(filepath)
        advanced_features["is_executable"] = 1
        advanced_features["num_sections"] = len(pe.sections)
        
        # Mengecek apakah ada bagian (section) file yang entropinya sangat ekstrem (dibungkus)
        for section in pe.sections:
            if section.get_entropy() > 7.5:
                advanced_features["has_high_entropy_section"] = 1
                break

        # Mengecek apakah aplikasi ini meminjam API berbahaya dari Windows
        bad_apis = [b"VirtualAlloc", b"CreateRemoteThread", b"InternetOpen", b"WriteProcessMemory"]
        api_count = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name and any(bad in imp.name for bad in bad_apis):
                        api_count += 1
        
        advanced_features["suspicious_api_count"] = api_count

    except pefile.PEFormatError:
        # File ini bukan aplikasi exe/dll (misal: cuma file teks)
        pass
    except Exception as e:
        pass
        
    return advanced_features

# ==== CARA UJI COBA ====
if __name__ == "__main__":
    import sys
    
    print("\n--- DETECTOR FORENSIK LANJUT ---")
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        print(f"Menganalisis: {target_file}")
        
        if os.path.exists(target_file):
            hasil = extract_advanced_features(target_file)
            print("Hasil Fitur Baru (Siap disuapkan ke CSV Machine Learning):")
            for k, v in hasil.items():
                print(f"  - {k}: {v}")
        else:
            print("File tidak ditemukan.")
    else:
        print("Jalankan skrip ini dengan memasukkan nama file di sebelahnya.")
        print("Contoh: python advanced_extractor_demo.py virus.exe")
