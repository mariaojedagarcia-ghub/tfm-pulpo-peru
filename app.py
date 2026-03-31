import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ─── Configuración de la página ───
st.set_page_config(
    page_title="Predicción Pulpo Perú",
    page_icon="🐙",
    layout="wide"
)

# ─── Estilos CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .detail {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    .info-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Título ───
st.title("🐙 Predicción de Desembarques de Pulpo — Perú")
st.markdown(
    "Modelo Ridge Regression entrenado con índices climáticos del Pacífico "
    "(Niño 1+2 y SOI) y datos históricos de capturas."
)

# ─── Carga de datos y modelo ───
@st.cache_resource
def load_assets():
    """Carga el modelo, el scaler y los datos preparados."""
    model = joblib.load('modelo_ridge_final.pkl')
    scaler = joblib.load('scaler_ridge.pkl')
    features = joblib.load('feature_names.pkl')
    df = pd.read_parquet('datos_modelo.parquet')
    return model, scaler, features, df

try:
    model, scaler, feature_names, df = load_assets()

    # ─── Barra lateral: entrada de datos ───
    st.sidebar.header("📊 Configuración de Predicción")
    st.sidebar.markdown("Introduce los valores climáticos del mes que quieres predecir:")

    # Último registro conocido (para valores por defecto y lags)
    ultimo = df.iloc[-1]

    st.sidebar.subheader("Datos del mes a predecir")
    mes_pred = st.sidebar.selectbox(
        "Mes",
        options=list(range(1, 13)),
        format_func=lambda m: [
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ][m - 1],
        index=0
    )

    nino_input = st.sidebar.number_input(
        "Índice Niño 1+2 (°C)",
        value=float(ultimo['nino12']),
        step=0.1,
        format="%.2f",
        help="Temperatura absoluta en la región Niño 1+2. Fuente: NOAA."
    )

    soi_input = st.sidebar.number_input(
        "Índice SOI",
        value=float(ultimo['soi']),
        step=0.1,
        format="%.1f",
        help="Southern Oscillation Index. Fuente: BOM Australia."
    )

    desemb_ultimo = st.sidebar.number_input(
        "Desembarque del mes anterior (t)",
        value=float(ultimo['desembarques_t']),
        step=0.5,
        format="%.2f",
        help="Dato de captura del mes inmediatamente anterior."
    )

    # ─── Construir el vector de features ───
    # Los lags se toman de los últimos registros conocidos
    # (en una app real, se irían actualizando con cada mes nuevo)
    ejemplo = pd.DataFrame([{
        'nino12':           nino_input,
        'soi':              soi_input,
        'mes_sin':          np.sin(2 * np.pi * mes_pred / 12),
        'mes_cos':          np.cos(2 * np.pi * mes_pred / 12),
        'nino12_lag1':      ultimo['nino12'],
        'soi_lag1':         ultimo['soi'],
        'nino12_lag3':      ultimo['nino12_lag1'],
        'soi_lag3':         ultimo['soi_lag1'],
        'nino12_lag6':      ultimo['nino12_lag3'],
        'soi_lag6':         ultimo['soi_lag3'],
        'nino12_lag12':     ultimo['nino12_lag6'],
        'soi_lag12':        ultimo['soi_lag6'],
        'desembarque_lag1': desemb_ultimo,
    }])

    # Asegurar orden correcto de columnas
    ejemplo = ejemplo[feature_names]

    # ─── Predicción ───
    ejemplo_sc = scaler.transform(ejemplo)
    pred = model.predict(ejemplo_sc)[0]
    pred_final = max(0, pred)

    # ─── Panel de resultados ───
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicción de desembarque</h3>
            <p class="value">{pred_final:.1f} t</p>
            <p class="detail">MAPE del modelo: 15.3%</p>
        </div>
        """, unsafe_allow_html=True)

        # Rango estimado con el MAPE
        margen = pred_final * 0.153
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f59e0b;">
            <h3>Rango estimado (±15.3%)</h3>
            <p class="value" style="font-size: 1.4rem;">{max(0, pred_final - margen):.1f} — {pred_final + margen:.1f} t</p>
            <p class="detail">Intervalo basado en el error medio del modelo</p>
        </div>
        """, unsafe_allow_html=True)

        # Comparación con la media histórica
        media_hist = df['desembarques_t'].mean()
        diff_pct = ((pred_final - media_hist) / media_hist) * 100
        signo = "+" if diff_pct > 0 else ""
        color_diff = "#22c55e" if diff_pct > 0 else "#ef4444"

        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color_diff};">
            <h3>vs. media histórica ({media_hist:.1f} t)</h3>
            <p class="value" style="font-size: 1.4rem; color: {color_diff};">{signo}{diff_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gráfica histórica con la predicción marcada
        df_plot = df.copy()
        df_plot['fecha'] = pd.to_datetime(
            df_plot['año'].astype(str) + '-' + df_plot['mes'].astype(str).str.zfill(2) + '-01'
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot['fecha'],
            y=df_plot['desembarques_t'],
            mode='lines',
            name='Histórico',
            line=dict(color='#0ea5e9', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(14, 165, 233, 0.08)'
        ))

        # Marcar la predicción
        ultima_fecha = df_plot['fecha'].iloc[-1]
        fecha_pred = ultima_fecha + pd.DateOffset(months=1)

        fig.add_trace(go.Scatter(
            x=[fecha_pred],
            y=[pred_final],
            mode='markers',
            name=f'Predicción ({pred_final:.1f} t)',
            marker=dict(color='#f59e0b', size=12, symbol='diamond',
                        line=dict(width=2, color='white'))
        ))

        fig.update_layout(
            title="Evolución histórica de capturas",
            xaxis_title="Fecha",
            yaxis_title="Desembarque (t)",
            template="plotly_dark",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ─── Sección inferior: detalle de los datos de entrada ───
    st.divider()

    with st.expander("🔍 Detalle de las variables usadas en la predicción"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Variables climáticas**")
            detalle_clima = pd.DataFrame({
                'Variable': ['Niño 1+2 actual', 'SOI actual',
                             'Niño 1+2 (lag 1m)', 'SOI (lag 1m)',
                             'Niño 1+2 (lag 3m)', 'SOI (lag 3m)',
                             'Niño 1+2 (lag 6m)', 'SOI (lag 6m)',
                             'Niño 1+2 (lag 12m)', 'SOI (lag 12m)'],
                'Valor': [
                    f"{ejemplo['nino12'].values[0]:.2f}",
                    f"{ejemplo['soi'].values[0]:.1f}",
                    f"{ejemplo['nino12_lag1'].values[0]:.2f}",
                    f"{ejemplo['soi_lag1'].values[0]:.1f}",
                    f"{ejemplo['nino12_lag3'].values[0]:.2f}",
                    f"{ejemplo['soi_lag3'].values[0]:.1f}",
                    f"{ejemplo['nino12_lag6'].values[0]:.2f}",
                    f"{ejemplo['soi_lag6'].values[0]:.1f}",
                    f"{ejemplo['nino12_lag12'].values[0]:.2f}",
                    f"{ejemplo['soi_lag12'].values[0]:.1f}",
                ]
            })
            st.dataframe(detalle_clima, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Variables estacionales y autorregresivas**")
            detalle_otro = pd.DataFrame({
                'Variable': ['Mes (seno)', 'Mes (coseno)', 'Desembarque mes anterior'],
                'Valor': [
                    f"{ejemplo['mes_sin'].values[0]:.4f}",
                    f"{ejemplo['mes_cos'].values[0]:.4f}",
                    f"{ejemplo['desembarque_lag1'].values[0]:.2f} t"
                ]
            })
            st.dataframe(detalle_otro, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ Sobre el modelo:</strong> Ridge Regression (α=10) con 13 variables de entrada.
        Entrenado con datos de 1997 a 2025. Los lags climáticos se toman de los últimos meses
        disponibles en el dataset. Para predicciones reales, actualiza los índices con los
        datos más recientes de la NOAA y el BOM.
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"Archivo no encontrado: {e}")
    st.info(
        "Asegúrate de que estos 4 archivos estén en la misma carpeta que `app.py`:\n"
        "- `modelo_ridge_final.pkl`\n"
        "- `scaler_ridge.pkl`\n"
        "- `feature_names.pkl`\n"
        "- `datos_modelo.parquet`"
    )
except Exception as e:
    st.error(f"Error inesperado: {e}")
