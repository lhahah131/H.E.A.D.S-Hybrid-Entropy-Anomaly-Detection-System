from fastapi import APIRouter, Depends
from api.auth import verify_api_key
import os

router = APIRouter()

@router.get("/status",)
async def get_status(key: str = Depends(verify_api_key)):
    mdoel_path = "Model/iso_v1_production.pkl"
    return{
        "sistem": "H.E.A.D.S. v1.5",
        "model" : "Aktif✅",
        "model_tersedia": os.path.exists(model_path)
    }
    