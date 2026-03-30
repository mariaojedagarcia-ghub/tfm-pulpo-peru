import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# Configuración de la página
st.set_page_config(page_title="Predicción Pulpo Perú", layout="wide")

st.title("Sistema de Predicción de Desembarques de Pulpo - Perú")
st.markdown("""
Esta aplicación utiliza un modelo de Machine Learning (Ridge Regression) para estimar 
las capturas de pulpo basadas en los índices climáticos del Pacífico (Niño 1+2 y SOI).
""")

# --- CARGA DE DATOS Y MODELO ---
@st.cache_resource
def load_assets():
    # Cambia estas rutas por las de tu proyecto
    data = pd.read_parquet('datos.parquet')
    model = joblib.load('modelo_ridge_final.pkl')
    return data, model

try:
    df, model = load_assets()
    
    # --- BARRA LATERAL: ENTRADA DE DATOS ---
    st.sidebar.header("Configuración de Predicción")
    st.sidebar.write("Introduce los valores climáticos actuales para predecir:")
    
    # Usamos los últimos valores conocidos como sugerencia
    last_nino = float(df['nino12'].iloc[-1])
    last_soi = float(df['soi'].iloc[-1])
    
    nino_input = st.sidebar.number_input("Índice Niño 1+2 actual", value=last_nino)
    soi_input = st.sidebar.number_input("Índice SOI actual", value=last_soi)

    # --- PREDICCIÓN ---
    # Nota: El modelo espera los lags que creamos. Para esta demo simple, 
    # usamos el valor actual como estimador del lag.
    features = [[nino_input, nino_input, nino_input, nino_input, soi_input, soi_input, soi_input, soi_input]]
    # (Ajusta 'features' según las columnas exactas que usaste en tu entrenamiento X_train)
    
    pred = model.predict(features)[0]
    pred_final = max(0, pred) # Aplicamos la solución al punto 2: evitar negativos

    # --- VISUALIZACIÓN ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Predicción Desembarque (t)", value=f"{pred_final:.2f} t")
        st.write("Margen de error (MAPE): 20.3%")
        
    with col2:
        # Gráfica histórica
        fig = px.line(df, x=df.index, y='desembarques_t', title="Evolución Histórica de Capturas")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.info("Asegúrate de que el modelo (.pkl) y el dataset (.parquet) estén en la misma carpeta que este script.")