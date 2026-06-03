from pathlib import Path

# Obtiene la ruta raíz del proyecto dinámicamente
ROOT_DIR = Path(__file__).resolve().parents[1]

# Rutas de carpetas de datos
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"

# Rutas de reportes
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"

# Archivos específicos
RAW_DATA_PATH = RAW_DATA_DIR / "dataset_sintetico_raw.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "dataset_sintetico_processed.csv"
RL_SIMULATED_DATA_PATH = PROCESSED_DATA_DIR / "agente_rl_simulacion.csv" 

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Dengue_Wolbachia_Experiment"
REGISTERED_MODEL_NAME = "modelo_wolbachia"

# Hiperparámetros globales
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100