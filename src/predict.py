import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    return mlflow.pyfunc.load_model(model_uri)

def predict_liberacion():
    model = load_model()

    input_df = pd.DataFrame([{
        "temperatura_media": 34.0,
        "humedad_relativa": 75.0,
        "casos_previos": 100,
    }])

    prediction = model.predict(input_df)

    print(" Mosquitos a liberar (Predicción):", int(prediction[0]))

if __name__ == "__main__":
    predict_liberacion()