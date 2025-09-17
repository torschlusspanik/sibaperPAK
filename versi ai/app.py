from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import os

MODEL_PATH = os.getenv('MODEL_PATH','model/sibaper_total_pipeline.joblib')

app = FastAPI(title="SIBAPER ML API", version="1.0.0",
              description="API untuk prediksi estimasi total pajak kendaraan bermotor (SIBAPER)")

# CORS so your HTML can call it from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    jns_kend: str = Field(..., description="R2 atau R4")
    pkb_pokok: int = Field(..., ge=0)
    usia_kend: int = Field(..., ge=0, le=50)
    tunggakan_tahun: int = Field(..., ge=0, le=10)

class PredictOut(BaseModel):
    predicted_total: int
    currency: str = "IDR"
    model_version: str = "sibaper_total_pipeline.joblib"

# Load model once
pipe = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status":"ok"}

import pandas as pd

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    # Buat DataFrame sesuai kolom training
    X = pd.DataFrame([{
        "jns_kend": inp.jns_kend,
        "pkb_pokok": inp.pkb_pokok,
        "usia_kend": inp.usia_kend,
        "tunggakan_tahun": inp.tunggakan_tahun
    }])

    y = pipe.predict(X)[0]
    return PredictOut(predicted_total=int(y))
