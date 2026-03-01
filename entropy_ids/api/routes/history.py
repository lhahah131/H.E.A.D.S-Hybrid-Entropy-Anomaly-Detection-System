import os
import csv
from fastapi import APIRouter, Depends
from api.auth import verify_api_key

router = APIRouter()
LOG_PATH = "logs/scan_history.log"

@router.get("/history")
async def get_history(key: str = Depends(verify_api_key)):
    if not os.path.exists(LOG_PATH):
        return {"data": [], "pesan": "Belum ada riwayat scan."}

    riwayat = []
    with open(LOG_PATH, "r") as f:
        for baris in f.readlines()[-50]:
            riwayat.append(baris.strip())
    
    return {"data": riwayat}
