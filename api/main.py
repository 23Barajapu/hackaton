"""
FastAPI endpoint untuk prediksi gagal panen.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import predict as pred_module
import json

app = FastAPI(
    title="Harvest Failure Prediction API",
    description="API untuk prediksi gagal panen menggunakan model GRU",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    region: str
    start_date: Optional[str] = None
    use_csv: bool = True

class PredictionResponse(BaseModel):
    region: str
    probability: float
    threshold: float
    prediction: str
    risk_level: str
    confidence: str
    reasons: list = []
    mitigation_recommendations: list = []
    weather_forecast: dict = {}

@app.get("/")
async def root():
    """Endpoint root untuk health check."""
    return {
        "message": "Harvest Failure Prediction API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Coba load model untuk memastikan semuanya berfungsi
        model, scaler, config = pred_module.load_model_and_artifacts()
        return {
            "status": "healthy",
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_harvest_failure(request: PredictionRequest):
    """
    Memprediksi kemungkinan gagal panen untuk suatu wilayah.
    
    Args:
        request: PredictionRequest dengan region, start_date (opsional), dan use_csv
    
    Returns:
        PredictionResponse dengan hasil prediksi
    """
    try:
        result = pred_module.predict_harvest_failure(
            region_name=request.region,
            start_date=request.start_date,
            use_csv=request.use_csv
        )
        
        # Jika ada error dalam result
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return PredictionResponse(**result)
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model tidak ditemukan. Pastikan model sudah dilatih terlebih dahulu. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saat melakukan prediksi: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(regions: list[str], use_csv: bool = True):
    """
    Memprediksi untuk beberapa wilayah sekaligus.
    
    Args:
        regions: List nama kabupaten/kota
        use_csv: Jika True, gunakan data CSV lokal
    
    Returns:
        List hasil prediksi untuk setiap wilayah
    """
    try:
        results = pred_module.predict_batch(regions, use_csv=use_csv)
        return {
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saat melakukan batch prediction: {str(e)}"
        )

@app.get("/regions")
async def get_available_regions():
    """
    Mendapatkan daftar wilayah yang tersedia dalam data.
    """
    try:
        import data_processing as dp
        df_harvest, _ = dp.load_data_from_csv()
        regions = sorted(df_harvest['Kabupaten/Kota'].unique().tolist())
        return {
            "regions": regions,
            "total": len(regions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saat mengambil daftar wilayah: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

