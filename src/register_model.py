import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError("No existe el experimento")

    # Buscamos el mejor modelo basado en la métrica R2 de Mayor a Menor (DESC)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No hay runs disponibles")

    best_run = runs[0]
    run_id = best_run.info.run_id

    # Ojo: En train.py guardamos el artefacto como "model"
    model_uri = f"runs:/{run_id}/model"

    # Registramos el modelo en el Registry
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    # Le ponemos la corona de "Campeón"
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=result.version
    )

    print("✅ ¡Mejor Modelo registrado correctamente!")
    print(f"🏆 Run ID (Ganador): {run_id}")
    print(f"📦 Modelo: {REGISTERED_MODEL_NAME}")
    print(f"🔄 Versión: {result.version}")
    print("🏷️ Alias: champion")

if __name__ == "__main__":
    register_best_model()