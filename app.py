import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# ─── Configuración de la página ───
st.set_page_config(
    page_title="TFM · Predicción Pulpo Perú — María Ojeda García",
    page_icon="icon_pulpo.png",
    layout="wide"
)

# ─── Estilos CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.78rem;
        margin: 0 0 0.2rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .detail {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.2rem;
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
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  CABECERA
# ═══════════════════════════════════════════
col_icon, col_title = st.columns([0.06, 0.94])
with col_icon:
    st.image("icon_pulpo.png", width=60)
with col_title:
    st.title("Predicción de Desembarques de Pulpo — Perú")

st.markdown(
    "**Trabajo Fin de Máster** · María Ojeda García  \n"
    "Esta herramienta estima cuánto pulpo llegará a los puertos peruanos el próximo mes, "
    "a partir de los índices climáticos del Pacífico (Niño 1+2 y SOI) y del historial de capturas."
)

# ═══════════════════════════════════════════
#  CARGA DE DATOS Y MODELO
# ═══════════════════════════════════════════
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

@st.cache_resource
def load_assets():
    model = joblib.load('modelo_ridge_final.pkl')
    scaler = joblib.load('scaler_ridge.pkl')
    features = joblib.load('feature_names.pkl')
    df = pd.read_parquet('datos_modelo.parquet')
    df_dep = None
    if os.path.exists('datos_departamento.parquet'):
        df_dep = pd.read_parquet('datos_departamento.parquet')
    return model, scaler, features, df, df_dep


try:
    model, scaler, feature_names, df, df_dep = load_assets()

    # Preparar fechas
    df_plot = df.copy()
    df_plot['fecha'] = pd.to_datetime(
        df_plot['año'].astype(int).astype(str) + '-' +
        df_plot['mes'].astype(int).astype(str).str.zfill(2) + '-01'
    )

    ultimo = df.iloc[-1]
    ultimo_fecha = f"{MESES[int(ultimo['mes']) - 1]} {int(ultimo['año'])}"
    ultimo_desemb = float(ultimo['desembarques_t'])

    # ═══════════════════════════════════════
    #  BARRA LATERAL — PREDICCIÓN
    # ═══════════════════════════════════════
    st.sidebar.image("icon_pulpo.png", width=50)
    st.sidebar.header("Predecir próximo mes")
    st.sidebar.caption(
        "Introduce los valores climáticos publicados por la NOAA "
        "para estimar el desembarque del mes que quieras."
    )

    mes_pred = st.sidebar.selectbox(
        "Mes a predecir",
        options=list(range(1, 13)),
        format_func=lambda m: MESES[m - 1],
        index=0
    )

    nino_input = st.sidebar.number_input(
        "Índice Niño 1+2 (°C)",
        value=float(ultimo['nino12']),
        step=0.1,
        format="%.2f",
        help="Temperatura del mar en la región Niño 1+2 (frente a la costa peruana). Fuente: NOAA."
    )

    soi_input = st.sidebar.number_input(
        "Índice SOI",
        value=float(ultimo['soi']),
        step=0.1,
        format="%.1f",
        help="Diferencia de presión atmosférica Tahití–Darwin. Fuente: BOM Australia."
    )

    desemb_ultimo = st.sidebar.number_input(
        "Desembarque del mes anterior (t)",
        value=float(ultimo['desembarques_t']),
        step=0.5,
        format="%.2f",
        help="Captura total registrada el mes anterior (en toneladas)."
    )

    # Construir features
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
    ejemplo = ejemplo[feature_names]

    # Predicción
    ejemplo_sc = scaler.transform(ejemplo)
    pred = model.predict(ejemplo_sc)[0]
    pred_final = max(0, pred)
    margen = pred_final * 0.153

    # ═══════════════════════════════════════
    #  PESTAÑAS PRINCIPALES
    # ═══════════════════════════════════════
    tab1, tab2 = st.tabs(["Vista general", "Por departamento"])

    # ─────────────────────────────────────
    #  PESTAÑA 1: VISTA GENERAL
    # ─────────────────────────────────────
    with tab1:

        # --- Fila de tarjetas resumen ---
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Último dato registrado</h3>
                <p class="value">{ultimo_desemb:.1f} t</p>
                <p class="detail">{ultimo_fecha}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <h3>Predicción próximo mes</h3>
                <p class="value">{pred_final:.1f} t</p>
                <p class="detail">{MESES[mes_pred - 1]} · rango: {max(0, pred_final - margen):.0f}–{pred_final + margen:.0f} t</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            media_hist = df['desembarques_t'].mean()
            diff_pct = ((pred_final - media_hist) / media_hist) * 100
            signo = "+" if diff_pct > 0 else ""
            color_diff = "#22c55e" if diff_pct > 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color_diff};">
                <h3>vs. media histórica</h3>
                <p class="value" style="color: {color_diff};">{signo}{diff_pct:.1f}%</p>
                <p class="detail">Media: {media_hist:.1f} t/mes</p>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            nino_val = float(ultimo['nino12'])
            if nino_val > 25:
                estado_clima = "El Niño (cálido)"
                color_clima = "#ef4444"
            elif nino_val < 22:
                estado_clima = "La Niña (frío)"
                color_clima = "#3b82f6"
            else:
                estado_clima = "Neutro"
                color_clima = "#a3a3a3"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color_clima};">
                <h3>Condición climática actual</h3>
                <p class="value" style="font-size: 1.3rem; color: {color_clima};">{estado_clima}</p>
                <p class="detail">Niño 1+2: {nino_val:.1f} °C · SOI: {float(ultimo['soi']):.1f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # --- Gráfica histórica ---
        st.subheader("Evolución de las capturas mensuales")
        st.caption(
            "Cada punto es un mes. El diamante naranja marca la predicción del modelo "
            "para el mes seleccionado en la barra lateral."
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot['fecha'],
            y=df_plot['desembarques_t'],
            mode='lines',
            name='Desembarques reales',
            line=dict(color='#0ea5e9', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(14, 165, 233, 0.07)',
            hovertemplate='%{x|%b %Y}: %{y:.1f} t<extra></extra>'
        ))

        # Predicción
        ultima_fecha_dt = df_plot['fecha'].iloc[-1]
        fecha_pred = ultima_fecha_dt + pd.DateOffset(months=1)

        fig.add_trace(go.Scatter(
            x=[fecha_pred],
            y=[pred_final],
            mode='markers',
            name=f'Predicción: {pred_final:.1f} t',
            marker=dict(color='#f59e0b', size=14, symbol='diamond',
                        line=dict(width=2, color='white')),
            hovertemplate='Predicción: %{y:.1f} t<extra></extra>'
        ))

        # Barra de rango
        fig.add_trace(go.Scatter(
            x=[fecha_pred, fecha_pred],
            y=[max(0, pred_final - margen), pred_final + margen],
            mode='lines',
            line=dict(color='#f59e0b', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="",
            yaxis_title="Toneladas",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Gráfica de estacionalidad ---
        st.subheader("Patrón estacional de la pesca")
        st.caption(
            "¿En qué meses del año se pesca más pulpo? "
            "El gráfico muestra el promedio histórico de capturas por mes."
        )

        estacional = df.groupby('mes')['desembarques_t'].agg(['mean', 'std']).reset_index()
        estacional['mes_nombre'] = estacional['mes'].apply(lambda m: MESES[int(m) - 1][:3])

        fig_est = go.Figure()
        fig_est.add_trace(go.Bar(
            x=estacional['mes_nombre'],
            y=estacional['mean'],
            marker_color='#0ea5e9',
            name='Media mensual',
            hovertemplate='%{x}: %{y:.1f} t<extra></extra>'
        ))
        fig_est.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="",
            yaxis_title="Toneladas (media)",
            showlegend=False
        )
        st.plotly_chart(fig_est, use_container_width=True)

        # --- Cómo funciona el modelo ---
        with st.expander("¿Cómo funciona este modelo?"):
            st.markdown("""
El modelo utiliza una **regresión Ridge** (un tipo de regresión lineal diseñada para
evitar el sobreajuste) para estimar cuánto pulpo llegará a puerto el próximo mes.

Se alimenta de **13 variables**:

- **Índice Niño 1+2**: temperatura del mar justo enfrente de la costa peruana.
  Cuando sube mucho (El Niño), suele afectar a la pesca.
- **Índice SOI**: mide la presión atmosférica en el Pacífico. Complementa al Niño 1+2
  porque capta lo que pasa en la atmósfera, no solo en el agua.
- **Retardos climáticos** (1, 3, 6 y 12 meses atrás): el clima de meses anteriores
  influye en la disponibilidad futura de pulpo, ya que afecta al crecimiento
  de los individuos jóvenes.
- **Desembarque del mes anterior**: el dato más determinante. Recoge de forma
  implícita el esfuerzo pesquero, las vedas vigentes y las condiciones locales.
- **Estacionalidad** (seno y coseno del mes): la pesca de pulpo tiene un ritmo
  anual que el modelo necesita conocer.

**Rendimiento en test** (últimos 24 meses): R² = 0.831, MAE = 22.9 t, MAPE = 20.3%.
            """)

        # --- Detalle técnico (colapsado) ---
        with st.expander("Detalle de las variables usadas en esta predicción"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Variables climáticas**")
                detalle_clima = pd.DataFrame({
                    'Variable': ['Niño 1+2 actual', 'SOI actual',
                                 'Niño 1+2 (hace 1 mes)', 'SOI (hace 1 mes)',
                                 'Niño 1+2 (hace 3 meses)', 'SOI (hace 3 meses)',
                                 'Niño 1+2 (hace 6 meses)', 'SOI (hace 6 meses)',
                                 'Niño 1+2 (hace 12 meses)', 'SOI (hace 12 meses)'],
                    'Valor': [
                        f"{ejemplo['nino12'].values[0]:.2f} °C",
                        f"{ejemplo['soi'].values[0]:.1f}",
                        f"{ejemplo['nino12_lag1'].values[0]:.2f} °C",
                        f"{ejemplo['soi_lag1'].values[0]:.1f}",
                        f"{ejemplo['nino12_lag3'].values[0]:.2f} °C",
                        f"{ejemplo['soi_lag3'].values[0]:.1f}",
                        f"{ejemplo['nino12_lag6'].values[0]:.2f} °C",
                        f"{ejemplo['soi_lag6'].values[0]:.1f}",
                        f"{ejemplo['nino12_lag12'].values[0]:.2f} °C",
                        f"{ejemplo['soi_lag12'].values[0]:.1f}",
                    ]
                })
                st.dataframe(detalle_clima, use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Estacionalidad y componente autorregresivo**")
                detalle_otro = pd.DataFrame({
                    'Variable': ['Mes (componente seno)', 'Mes (componente coseno)',
                                 'Desembarque del mes anterior'],
                    'Valor': [
                        f"{ejemplo['mes_sin'].values[0]:.4f}",
                        f"{ejemplo['mes_cos'].values[0]:.4f}",
                        f"{ejemplo['desembarque_lag1'].values[0]:.2f} t"
                    ]
                })
                st.dataframe(detalle_otro, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────
    #  PESTAÑA 2: POR DEPARTAMENTO
    # ─────────────────────────────────────
    with tab2:

        if df_dep is not None:
            st.subheader("Distribución de capturas por departamento")
            st.caption(
                "¿Dónde se pesca más pulpo a lo largo de la costa peruana? "
                "Los datos proceden del IMARPE y cubren todos los departamentos costeros."
            )

            dep_total = (
                df_dep.groupby('departamento')['desembarque_t']
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_dep = px.bar(
                dep_total,
                x='desembarque_t',
                y='departamento',
                orientation='h',
                labels={'desembarque_t': 'Toneladas totales', 'departamento': ''},
                template='plotly_dark',
                color='desembarque_t',
                color_continuous_scale='Blues'
            )
            fig_dep.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dep, use_container_width=True)

            # Evolución temporal por departamento (top 5)
            st.subheader("Evolución temporal por departamento")
            st.caption("Desembarques anuales de los 5 departamentos con más capturas.")

            top_deps = dep_total['departamento'].head(5).tolist()
            dep_anual = (
                df_dep[df_dep['departamento'].isin(top_deps)]
                .groupby(['anio', 'departamento'])['desembarque_t']
                .sum()
                .reset_index()
            )

            fig_dep_evol = px.line(
                dep_anual,
                x='anio',
                y='desembarque_t',
                color='departamento',
                labels={'desembarque_t': 'Toneladas', 'anio': 'Año', 'departamento': 'Departamento'},
                template='plotly_dark'
            )
            fig_dep_evol.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_dep_evol, use_container_width=True)

        else:
            st.info(
                "**Datos por departamento no disponibles.**\n\n"
                "Para activar esta sección, añade esta celda al final del notebook:\n\n"
                "```python\n"
                "dep_mensual = (\n"
                "    df_pulpo_raw\n"
                "    .groupby(['anio', 'mes', 'departamento'])['desembarque_kg']\n"
                "    .sum()\n"
                "    .reset_index()\n"
                ")\n"
                "dep_mensual['desembarque_t'] = dep_mensual['desembarque_kg'] / 1000\n"
                "dep_mensual.to_parquet(OUTPUT + 'datos_departamento.parquet', index=False)\n"
                "```\n\n"
                "Después, sube `datos_departamento.parquet` junto a los demás archivos."
            )

    # ═══════════════════════════════════════
    #  PIE DE PÁGINA
    # ═══════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.82rem; "
        "padding: 0.5rem 0 1rem;'>"
        "TFM — Predicción de desembarques de pulpo en Perú mediante variables climáticas<br>"
        "María Ojeda García · Máster en Big Data · 2025<br>"
        "Fuentes: IMARPE · NOAA (CPC) · BOM Australia"
        "</div>",
        unsafe_allow_html=True
    )

except FileNotFoundError as e:
    st.error(f"Archivo no encontrado: {e}")
    st.info(
        "Asegúrate de que estos archivos estén en la misma carpeta que `app.py`:\n"
        "- `modelo_ridge_final.pkl`\n"
        "- `scaler_ridge.pkl`\n"
        "- `feature_names.pkl`\n"
        "- `datos_modelo.parquet`\n"
        "- `icon_pulpo.png`\n"
        "- (opcional) `datos_departamento.parquet`"
    )
except Exception as e:
    st.error(f"Error inesperado: {e}")
