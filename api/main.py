import time
from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    MONITORING_DIR,
)

# Definimos las columnas tal como las entrenamos
FEATURE_COLUMNS = ["temperatura_media", "humedad_relativa", "casos_previos"]

app = FastAPI(title="API de Control Vectorial Dengue - Wolbachia")

# =========================
# MÉTRICAS PROMETHEUS (Para Grafana)
# =========================

PREDICTION_COUNT = Counter(
    "wolbachia_prediction_count",
    "Número total de predicciones de liberación de mosquitos"
)

PREDICTION_LATENCY = Histogram(
    "wolbachia_prediction_latency_seconds",
    "Latencia de predicción del modelo en segundos"
)

# =========================
# MODELO GLOBAL
# =========================

model = None

# =========================
# REQUEST MODEL
# =========================

class PredictionRequest(BaseModel):
    temperatura_media: float
    humedad_relativa: float
    casos_previos: int

# =========================
# STARTUP
# =========================

@app.on_event("startup")
def load_champion_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Modelo Champion cargado correctamente")

# =========================
# ROOT & HEALTH CHECK
# =========================

@app.get("/")
def root():
    return {"message": "Dengue Wolbachia API activa"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME,
        "alias": "champion",
    }

# =========================
# PREDICCIÓN
# =========================

@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()

    if model is None:
        return {"error": "El modelo todavía no ha sido cargado"}

    # Crear dataframe garantizando el orden exacto de columnas
    input_df = pd.DataFrame([{
        "temperatura_media": float(request.temperatura_media),
        "humedad_relativa": float(request.humedad_relativa),
        "casos_previos": int(request.casos_previos),
    }])
    input_df = input_df[FEATURE_COLUMNS]

    # Predicción
    prediction = int(model.predict(input_df)[0])
    latency = time.time() - start_time

    # Actualizar Métricas Prometheus
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)

    # Logging para Monitoreo de Data Drift
    log_prediction(input_df=input_df, prediction=prediction, latency=latency)

    return {
        "mosquitos_recomendados": prediction,
        "message": "Cálculo de dosis adaptativa completado",
        "latency_seconds": round(latency, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }

# =========================
# MÉTRICAS PROMETHEUS
# =========================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# =========================
# LOG PREDICCIONES
# =========================

def log_prediction(input_df: pd.DataFrame, prediction: int, latency: float):
    log_file = MONITORING_DIR / "api_predictions_log.csv"
    row = input_df.copy()
    row["mosquitos_recomendados"] = prediction
    row["latency_seconds"] = latency
    row["timestamp"] = datetime.utcnow().isoformat()

    if log_file.exists():
        row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        row.to_csv(log_file, index=False)