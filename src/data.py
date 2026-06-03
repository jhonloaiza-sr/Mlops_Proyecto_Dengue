import pandas as pd
import numpy as np
import os
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, RANDOM_STATE

def generate_raw_data():
    """Genera los datos sintéticos con 'imperfecciones' si no existen."""
    if not RAW_DATA_PATH.exists():
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Generando datos sintéticos RAW de Cali...")
        np.random.seed(RANDOM_STATE)
        n_samples = 500

        temperatura = np.random.uniform(24.0, 32.0, n_samples)
        humedad = np.random.uniform(60.0, 90.0, n_samples)
        casos_previos = np.random.randint(10, 100, n_samples)

        mosquitos = (temperatura * 1000) + (casos_previos * 200) - (humedad * 100) + np.random.normal(0, 1000, n_samples)

        # Creamos los datos con nombres feos y un ID inútil (para simular el mundo real)
        df = pd.DataFrame({
            "ID Zona": range(1, n_samples + 1),
            "Temperatura (Media)": temperatura,
            "Humedad Relativa": humedad,
            "Casos Previos": casos_previos,
            "Mosquitos a Liberar": mosquitos
        })
        df.to_csv(RAW_DATA_PATH, index=False)

def load_raw_data() -> pd.DataFrame:
    """Carga los datos raw. Si no existen, los genera primero."""
    generate_raw_data()
    return pd.read_csv(RAW_DATA_PATH)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia los nombres de las columnas (Igual a la lógica del profe)."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df

def preprocess_data() -> pd.DataFrame:
    """Pipeline de preprocesamiento principal."""
    df = load_raw_data()
    df = clean_column_names(df)

    # Eliminamos el ID tal como el profe elimina el 'customerid'
    if "id_zona" in df.columns:
        df = df.drop(columns=["id_zona"])

    # Validamos que las columnas existan
    expected_columns = ["temperatura_media", "humedad_relativa", "casos_previos", "mosquitos_a_liberar"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}.")

    # Guardamos en processed
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df

if __name__ == "__main__":
    processed = preprocess_data()
    print(" Transformación completada. Muestra de los datos procesados:")
    print(processed.head())
    print(processed.info())