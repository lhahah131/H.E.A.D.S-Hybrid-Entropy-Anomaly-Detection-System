from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import scan, history, status


app = FastAPI(
    title="H.E.A.D.S. API",
    description="Next-Gen Antivirus - Hybeid Entopy Anomaly Detection System",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/api/v1", tags=["Scan"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])
app.include_router(status.router, prefix="/api/v1", tags=["Status"])

@app.get("/")
def root():
    return {"pesan": "H.E.A.D.S. API aktif. Buka /docs untuk dokumentasi."}