# 🦟 MLOps Pipeline: Sistema Dinámico de Control Vectorial (Dengue - Wolbachia)

Este repositorio contiene la implementación de una arquitectura MLOps completa orientada a la salud pública. El sistema simula las decisiones de un Agente de Aprendizaje por Refuerzo (RL) para recomendar dinámicamente la liberación de mosquitos infectados con *Wolbachia*, optimizando los recursos logísticos frente a variables climáticas y epidemiológicas en la ciudad de Cali.

## 👥 Equipo de Trabajo
*   **Juan Manuel Cajigas Eraso**
*   **Jonathan Giraldo Diaz Ortega**
*   **Eliphas Levi Arias**
*   **Jhon Stiven Loaiza**

---

## 🏗️ Arquitectura del Proyecto

El proyecto sigue las mejores prácticas de la industria en MLOps, garantizando reproducibilidad, trazabilidad y monitoreo en tiempo real:

1.  **Data Ingestion & Pipeline (`src/data.py`):** Extracción y transformación de datos sintéticos simulando condiciones climáticas.
2.  **Experiment Tracking (`src/train.py`):** Entrenamiento competitivo de múltiples modelos supervisados y registro de artefactos/métricas usando **MLflow**.
3.  **Model Registry (`src/register_model.py`):** Selección automatizada del mejor modelo basado en métricas (Ej. R2) y asignación del tag `champion` en producción.
4.  **Model Serving (`api/main.py`):** Despliegue del modelo campeón mediante una API REST en **FastAPI**.
5.  **User Interface (`app/streamlit_app.py`):** Dashboard interactivo desarrollado en **Streamlit** para la toma de decisiones in-silico.
6.  **Monitoring & Observability:** Exposición de métricas de latencia y tráfico vía **Prometheus** (`/metrics`) y logs locales para detección de *Data Drift* (`data/monitoring/`).

---

## 📂 Estructura del Repositorio

```text
proyecto_mlops_dengue/
│
├── api/
│   └── main.py                  # Endpoints de FastAPI y métricas Prometheus
│
├── app/
│   └── streamlit_app.py         # Interfaz de usuario (Dashboard)
│
├── data/
│   ├── raw/                     # Datos crudos
│   ├── processed/               # Datos transformados listos para ML
│   └── monitoring/              # Logs de predicciones en producción (Data Drift)
│
├── reports/
│   ├── figures/                 # Gráficas generadas automáticamente
│   └── metrics/                 # Reportes de métricas en CSV
│
├── src/
│   ├── config.py                # Variables globales y rutas
│   ├── data.py                  # Pipeline ETL
│   ├── rl_agent.py              # Simulación del agente RL y generación de reportes
│   ├── train.py                 # Entrenamiento de modelos y tracking en MLflow
│   ├── register_model.py        # Registro del modelo "champion"
│   └── predict.py               # Script de prueba de inferencia local
│
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación de auditoría
```

---

## 🚀 Guía de Reproducción (Auditoría)

Para replicar y auditar la arquitectura en un entorno local, siga estos pasos estrictamente en orden:

### 1. Preparación del Entorno
Clonar el repositorio y configurar el entorno virtual:

```bash
# Crear y activar entorno virtual (Windows)
python -m venv venv
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecución del Pipeline de Datos y Machine Learning
Ejecute los scripts como módulos para generar los artefactos y poblar el registro de modelos:

```bash
# Paso 1: Generar y procesar datos
python -m src.data

# Paso 2: Simular comportamiento del Agente RL y generar gráficas de desempeño
python -m src.rl_agent

# Paso 3: Entrenar modelos (LinearRegression, RandomForest, GradientBoosting) y subir a MLflow
python -m src.train

# Paso 4: Evaluar experimentos y coronar al mejor modelo como 'champion'
python -m src.register_model
```

*📌 **Nota de Auditoría:** Puede verificar los artefactos generados (matrices de error, curvas de aprendizaje) revisando las carpetas `reports/figures/` y `reports/metrics/`.*

### 3. Despliegue de Servicios MLOps
Para ver el sistema en funcionamiento, es necesario levantar 3 terminales independientes (asegúrese de tener el entorno virtual activado en cada una):

**Terminal 1: Servidor de MLflow (Tracking & Registry)**
```bash
mlflow ui
# Acceso: http://127.0.0.1:5000
```

**Terminal 2: API de Inferencia (FastAPI)**
```bash
uvicorn api.main:app --reload
# Acceso a Swagger UI: http://127.0.0.1:8000/docs
# Acceso a Métricas Prometheus: http://127.0.0.1:8000/metrics
```

**Terminal 3: Interfaz de Usuario (Streamlit)**
```bash
streamlit run app/streamlit_app.py
# Acceso: http://localhost:8501
```

---

## 📊 Monitoreo y Mantenimiento

Este proyecto simula un entorno listo para producción, por lo que incluye:

*   **Métricas en Tiempo Real:** La API expone un endpoint de Prometheus en `/metrics` que contabiliza `wolbachia_prediction_count` y mide la distribución de tiempos de respuesta en `wolbachia_prediction_latency_seconds`.
*   **Monitoreo de Data Drift:** Cada petición realizada a través del Streamlit o la API queda registrada en formato *append* dentro de `data/monitoring/api_predictions_log.csv` junto con su marca de tiempo (timestamp) y latencia.