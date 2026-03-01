from fastapi import APIRouter, Depends, UploadFile, File
from api.auth import verify_api_key
import shutil, os

router = APIRouter()
SANDBOX_DIR = "data/sandbox"

@router.post("/scan")
async def upload_dan_scan(
    file: UploadFile = File(...),
    key: str = Depends(verify_api_key)
):
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    tujuan = os.path.join(SANDBOX_DIR, file.filename)

    with open(tujuan, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "nama_file": file.filename,
        "pesan": "File sedang melakukan analisis oleh scanner. Cek /history untuk melihat hasilnya."
    }