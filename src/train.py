import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    FIGURES_DIR,
    METRICS_DIR,
)

def load_processed_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "No existe el dataset procesado. Ejecuta primero: python -m src.data"
        )
    return pd.read_csv(PROCESSED_DATA_PATH)

def save_prediction_plot(y_test, y_pred, model_name: str):
    """Genera gráfica de Reales vs Predichos (El equivalente a la Matriz de Confusión para Regresión)"""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="blue")
    
    # Línea ideal (perfecta predicción)
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'r--', lw=2)
    
    plt.title(f"Reales vs Predichos - {model_name}")
    plt.xlabel("Mosquitos Reales (Optimizados)")
    plt.ylabel("Mosquitos Predichos")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"actual_vs_predicted_{model_name}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test):
    # Iniciamos el run en MLflow con el nombre del modelo
    with mlflow.start_run(run_name=model_name):
        # 1. Pipeline profesional (Escalado + Modelo)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # 2. Entrenamiento
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # 3. Métricas de Regresión
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 4. Registro en MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 5. Guardar reporte de métricas localmente
        report_df = pd.DataFrame({
            "Metric": ["MSE", "RMSE", "MAE", "R2"],
            "Value": [mse, rmse, mae, r2]
        })
        report_path = METRICS_DIR / f"regression_report_{model_name}.csv"
        report_df.to_csv(report_path, index=False)
        mlflow.log_artifact(str(report_path), artifact_path="metrics") # Lo sube a MLflow

        # 6. Generar gráfica y subir a MLflow
        plot_path = save_prediction_plot(y_test, y_pred, model_name)
        mlflow.log_artifact(str(plot_path), artifact_path="figures")

        # 7. Guardar el modelo con ejemplo de entrada (Fina coquetería)
        input_example = X_test.head(3)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
        )

        print(f" Modelo: {model_name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")

def train_models():
    # Aseguramos directorios
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Configuramos MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_processed_data()

    # Features (X) y Target (y)
    feature_columns = ["temperatura_media", "humedad_relativa", "casos_previos"]
    X = df[feature_columns]
    y = df["mosquitos_a_liberar"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Diccionario con 3 modelos para que compitan (Igual que el profe)
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    print("Iniciando entrenamiento y registro en MLflow...")
    for model_name, model in models.items():
        train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
    
    print(" ¡Todos los modelos han sido entrenados y registrados!")

if __name__ == "__main__":
    train_models()