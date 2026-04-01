from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Regresión Lineal - Predicción de Calificaciones")

# Carga el modelo entrenado
MODEL_PATH = os.path.join("models", "model.pkl")

model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    horas_estudio: float

class PredictResponse(BaseModel):
    horas_estudio: float
    calificacion_predicha: float

@app.get("/")
def root():
    return {"message": "API de Regresión Lineal activa"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = model.predict([[req.horas_estudio]])[0]
    return PredictResponse(
        horas_estudio=req.horas_estudio,
        calificacion_predicha=round(float(pred), 2)
    )