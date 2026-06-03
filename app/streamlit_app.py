import streamlit as st
import requests

# Configuración básica de la página
st.set_page_config(page_title="IA Control Dengue", page_icon="🦟", layout="centered")

# Título y descripción alineados a tu Canvas
st.title("🦟 Sistema MLOps: Control de Dengue con Wolbachia")
st.markdown("""
**Secretaría de Salud de Cali / World Mosquito Program**
Este dashboard simula la recomendación dinámica (basada en IA) para la liberación de mosquitos con *Wolbachia*, adaptándose a las condiciones climáticas y epidemiológicas actuales, evitando cronogramas estáticos y desperdicio biológico.
""")
st.divider()

st.header("Condiciones Actuales en Cali")

# Creamos columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with col1:
    temperatura = st.slider("Temperatura Media (°C)", min_value=20.0, max_value=35.0, value=28.0, step=0.5)
    humedad = st.slider("Humedad Relativa (%)", min_value=50.0, max_value=100.0, value=75.0, step=1.0)

with col2:
    casos = st.number_input("Casos de Dengue Previos (Zona)", min_value=0, max_value=500, value=50)

st.divider()

# Botón de predicción
if st.button(" Calcular dosis óptima de mosquitos", type="primary"):
    
    # Preparamos los datos para enviarlos a nuestra API (FastAPI)
    datos_api = {
        "temperatura_media": temperatura,
        "humedad_relativa": humedad,
        "casos_previos": casos
    }
    
    try:
        # Hacemos la petición POST a FastAPI
        # Nota: Asumimos que FastAPI corre en el puerto 8000
        respuesta = requests.post("http://127.0.0.1:8000/predict", json=datos_api)
        
        if respuesta.status_code == 200:
            resultado = respuesta.json()
            mosquitos = resultado["mosquitos_recomendados"]
            tiempo = resultado["latency_seconds"]
            
            st.success("✅ ¡Cálculo completado!")
            st.metric(label="🦟 Cantidad Recomendada a Liberar", value=f"{mosquitos:,} mosquitos")
            
            # AQUÍ MOSTRAMOS LA MÉTRICA DE TIEMPO
            st.caption(f"⏱️ **Métrica de Inferencia:** Tiempo de respuesta de la API: `{tiempo} ms`")
            
            st.info("💡 **Nota:** Esta recomendación dinámica permite lograr el >80% de inmunidad de rebaño reduciendo los costos logísticos.")
        else:
            st.error(f"Error en la API: {respuesta.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Error de conexión: Asegúrate de que la API (FastAPI) esté corriendo en otra terminal.")