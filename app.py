import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import requests
from io import StringIO
from datetime import datetime

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
        color: #94a3b8; font-size: 0.78rem; margin: 0 0 0.2rem 0;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .metric-card .detail { color: #64748b; font-size: 0.78rem; margin-top: 0.2rem; }
    .info-box {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 8px;
        padding: 1rem 1.2rem; font-size: 0.85rem; color: #94a3b8; margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  FUNCIONES DE DESCARGA NOAA
# ═══════════════════════════════════════════

@st.cache_data(ttl=86400)  # Cache 24h
def fetch_nino12_noaa():
    """Descarga los datos mensuales de Niño 1+2 de la NOAA (sstoi.indices)."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
        # Primera línea es cabecera; las demás son datos separados por espacios
        # Formato: YEAR MON NINO1+2 ANOM NINO3 ANOM NINO34 ANOM NINO4 ANOM
        data = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    nino12 = float(parts[2])
                    data.append({'año': year, 'mes': month, 'nino12': nino12})
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(data) if data else None
    except Exception:
        return None


@st.cache_data(ttl=86400)
def fetch_soi_noaa():
    """Descarga los datos mensuales de SOI de la NOAA."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
        # El archivo tiene tres bloques: anomalías, estandarizado, etc.
        # Buscamos la sección de datos estandarizados (SOI)
        # Formato ancho: YEAR JAN FEB MAR ... DEC
        data = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(float(parts[0]))
                    if year < 1950 or year > 2100:
                        continue
                    for m in range(1, 13):
                        val = float(parts[m])
                        if val < -90:  # Valor faltante (-999.9)
                            continue
                        data.append({'año': year, 'mes': m, 'soi': val})
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(data) if data else None
    except Exception:
        return None


def get_climate_data(año, mes, df_local, nino_noaa, soi_noaa):
    """
    Busca los datos climáticos para un mes concreto.
    Prioridad: 1) NOAA en línea, 2) dataset local.
    Devuelve (nino12, soi, fuente).
    """
    nino_val, soi_val = None, None
    fuente = "local"

    # Intentar NOAA
    if nino_noaa is not None:
        match = nino_noaa[(nino_noaa['año'] == año) & (nino_noaa['mes'] == mes)]
        if len(match) > 0:
            nino_val = float(match.iloc[0]['nino12'])
            fuente = "NOAA (en línea)"

    if soi_noaa is not None:
        match = soi_noaa[(soi_noaa['año'] == año) & (soi_noaa['mes'] == mes)]
        if len(match) > 0:
            soi_val = float(match.iloc[0]['soi'])

    # Fallback: dataset local
    if nino_val is None:
        match = df_local[(df_local['año'] == año) & (df_local['mes'] == mes)]
        if len(match) > 0:
            nino_val = float(match.iloc[0]['nino12'])
            soi_val = float(match.iloc[0]['soi'])
            fuente = "dataset local"

    return nino_val, soi_val, fuente


# ═══════════════════════════════════════════
#  CONSTANTES Y CARGA
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
    "Esta herramienta estima cuánto pulpo llegará a los puertos peruanos, "
    "a partir de los índices climáticos del Pacífico (Niño 1+2 y SOI) y del historial de capturas. "
    "Los datos climáticos se descargan automáticamente de la NOAA."
)

# ═══════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════

try:
    model, scaler, feature_names, df, df_dep = load_assets()

    # Preparar fechas del dataset
    df_plot = df.copy()
    df_plot['fecha'] = pd.to_datetime(
        df_plot['año'].astype(int).astype(str) + '-' +
        df_plot['mes'].astype(int).astype(str).str.zfill(2) + '-01'
    )

    ultimo = df.iloc[-1]
    ultimo_año = int(ultimo['año'])
    ultimo_mes = int(ultimo['mes'])

    # Descargar datos NOAA
    with st.spinner("Consultando datos climáticos de la NOAA..."):
        nino_noaa = fetch_nino12_noaa()
        soi_noaa = fetch_soi_noaa()

    noaa_ok = nino_noaa is not None and soi_noaa is not None

    # ═══════════════════════════════════════
    #  BARRA LATERAL — SELECTOR DE MES
    # ═══════════════════════════════════════

    st.sidebar.image("icon_pulpo.png", width=50)
    st.sidebar.header("Selecciona un mes")

    if noaa_ok:
        st.sidebar.success("Conectado a la NOAA")
        # El último mes disponible en NOAA
        ultimo_noaa_año = int(nino_noaa['año'].max())
        ultimo_noaa_row = nino_noaa[nino_noaa['año'] == ultimo_noaa_año]
        ultimo_noaa_mes = int(ultimo_noaa_row['mes'].max())
        st.sidebar.caption(
            f"Último dato disponible: {MESES[ultimo_noaa_mes - 1]} {ultimo_noaa_año}"
        )
        max_año = ultimo_noaa_año + 1
    else:
        st.sidebar.warning("Sin conexión a NOAA. Usando datos locales.", icon="⚠️")
        st.sidebar.caption(
            f"Último dato local: {MESES[ultimo_mes - 1]} {ultimo_año}"
        )
        max_año = ultimo_año + 1

    año_pred = st.sidebar.selectbox(
        "Año",
        options=list(range(max_año, 1996, -1)),
        index=0
    )

    mes_pred = st.sidebar.selectbox(
        "Mes",
        options=list(range(1, 13)),
        format_func=lambda m: MESES[m - 1],
        index=0
    )

    # ─── Buscar datos climáticos ───
    nino_val, soi_val, fuente_clima = get_climate_data(
        año_pred, mes_pred, df, nino_noaa, soi_noaa
    )

    # Si no hay datos para el mes seleccionado, usar los últimos disponibles
    datos_estimados = False
    if nino_val is None:
        # Buscar el último mes con datos
        if noaa_ok:
            last_row = nino_noaa.iloc[-1]
            nino_val = float(last_row['nino12'])
            soi_match = soi_noaa[
                (soi_noaa['año'] == int(last_row['año'])) &
                (soi_noaa['mes'] == int(last_row['mes']))
            ]
            soi_val = float(soi_match.iloc[0]['soi']) if len(soi_match) > 0 else float(ultimo['soi'])
            fuente_clima = f"Último dato NOAA ({MESES[int(last_row['mes']) - 1]} {int(last_row['año'])})"
        else:
            nino_val = float(ultimo['nino12'])
            soi_val = float(ultimo['soi'])
            fuente_clima = f"Último dato local ({MESES[ultimo_mes - 1]} {ultimo_año})"
        datos_estimados = True

    if soi_val is None:
        soi_val = float(ultimo['soi'])

    # Mostrar en la barra lateral
    st.sidebar.divider()
    st.sidebar.markdown("**Datos climáticos obtenidos:**")
    st.sidebar.metric("Niño 1+2", f"{nino_val:.2f} °C")
    st.sidebar.metric("SOI", f"{soi_val:.1f}")
    st.sidebar.caption(f"Fuente: {fuente_clima}")

    if datos_estimados:
        st.sidebar.warning(
            f"No hay datos climáticos para {MESES[mes_pred - 1]} {año_pred}. "
            f"Se usa el último valor disponible como aproximación.",
            icon="⚠️"
        )

    # ─── Buscar lags y desembarque anterior ───
    # Buscar la fila del mes seleccionado en el dataset
    fila_mes = df[(df['año'] == año_pred) & (df['mes'] == mes_pred)]

    if len(fila_mes) > 0:
        # El mes está en el dataset → usar sus lags reales
        row = fila_mes.iloc[0]
        lags = {
            'nino12_lag1': row['nino12_lag1'], 'soi_lag1': row['soi_lag1'],
            'nino12_lag3': row['nino12_lag3'], 'soi_lag3': row['soi_lag3'],
            'nino12_lag6': row['nino12_lag6'], 'soi_lag6': row['soi_lag6'],
            'nino12_lag12': row['nino12_lag12'], 'soi_lag12': row['soi_lag12'],
            'desembarque_lag1': row['desembarque_lag1'],
        }
        desemb_real = float(row['desembarques_t'])
        hay_dato_real = True
    else:
        # El mes NO está en el dataset → usar los últimos lags conocidos
        lags = {
            'nino12_lag1': ultimo['nino12'], 'soi_lag1': ultimo['soi'],
            'nino12_lag3': ultimo['nino12_lag1'], 'soi_lag3': ultimo['soi_lag1'],
            'nino12_lag6': ultimo['nino12_lag3'], 'soi_lag6': ultimo['soi_lag3'],
            'nino12_lag12': ultimo['nino12_lag6'], 'soi_lag12': ultimo['soi_lag6'],
            'desembarque_lag1': ultimo['desembarques_t'],
        }
        desemb_real = None
        hay_dato_real = False

    # ─── Construir features y predecir ───
    ejemplo = pd.DataFrame([{
        'nino12': nino_val,
        'soi': soi_val,
        'mes_sin': np.sin(2 * np.pi * mes_pred / 12),
        'mes_cos': np.cos(2 * np.pi * mes_pred / 12),
        **lags,
    }])
    ejemplo = ejemplo[feature_names]

    ejemplo_sc = scaler.transform(ejemplo)
    pred = model.predict(ejemplo_sc)[0]
    pred_final = max(0, pred)
    margen = pred_final * 0.153

    # ═══════════════════════════════════════
    #  PESTAÑAS
    # ═══════════════════════════════════════

    tab1, tab2 = st.tabs(["Vista general", "Por departamento"])

    with tab1:

        # --- Tarjetas resumen ---
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if hay_dato_real:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Dato real registrado</h3>
                    <p class="value">{desemb_real:.1f} t</p>
                    <p class="detail">{MESES[mes_pred - 1]} {año_pred}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Último dato registrado</h3>
                    <p class="value">{float(ultimo['desembarques_t']):.1f} t</p>
                    <p class="detail">{MESES[ultimo_mes - 1]} {ultimo_año}</p>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <h3>Predicción del modelo</h3>
                <p class="value">{pred_final:.1f} t</p>
                <p class="detail">{MESES[mes_pred - 1]} {año_pred} · rango: {max(0, pred_final - margen):.0f}–{pred_final + margen:.0f} t</p>
            </div>""", unsafe_allow_html=True)

        with c3:
            if hay_dato_real:
                error = abs(desemb_real - pred_final)
                error_pct = (error / desemb_real * 100) if desemb_real > 0 else 0
                color_err = "#22c55e" if error_pct < 20 else "#f59e0b" if error_pct < 35 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color_err};">
                    <h3>Error de la predicción</h3>
                    <p class="value" style="color: {color_err};">{error_pct:.1f}%</p>
                    <p class="detail">{error:.1f} t de diferencia</p>
                </div>""", unsafe_allow_html=True)
            else:
                media_hist = df['desembarques_t'].mean()
                diff_pct = ((pred_final - media_hist) / media_hist) * 100
                signo = "+" if diff_pct > 0 else ""
                color_diff = "#22c55e" if diff_pct > 0 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color_diff};">
                    <h3>vs. media histórica</h3>
                    <p class="value" style="color: {color_diff};">{signo}{diff_pct:.1f}%</p>
                    <p class="detail">Media: {media_hist:.1f} t/mes</p>
                </div>""", unsafe_allow_html=True)

        with c4:
            if nino_val > 25:
                estado, color_clima = "El Niño (cálido)", "#ef4444"
            elif nino_val < 22:
                estado, color_clima = "La Niña (frío)", "#3b82f6"
            else:
                estado, color_clima = "Neutro", "#a3a3a3"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {color_clima};">
                <h3>Condición climática</h3>
                <p class="value" style="font-size: 1.3rem; color: {color_clima};">{estado}</p>
                <p class="detail">Niño 1+2: {nino_val:.1f} °C · SOI: {soi_val:.1f}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # --- Gráfica histórica ---
        st.subheader("Evolución de las capturas mensuales")
        if hay_dato_real:
            st.caption(
                "La línea azul muestra los desembarques reales. "
                "El diamante naranja es lo que el modelo habría predicho para el mes seleccionado."
            )
        else:
            st.caption(
                "La línea azul muestra los desembarques reales. "
                "El diamante naranja es la predicción del modelo para el mes seleccionado."
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['fecha'], y=df_plot['desembarques_t'],
            mode='lines', name='Desembarques reales',
            line=dict(color='#0ea5e9', width=1.5),
            fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.07)',
            hovertemplate='%{x|%b %Y}: %{y:.1f} t<extra></extra>'
        ))

        # Punto de predicción
        try:
            fecha_pred = pd.Timestamp(year=año_pred, month=mes_pred, day=1)
        except:
            fecha_pred = df_plot['fecha'].iloc[-1] + pd.DateOffset(months=1)

        fig.add_trace(go.Scatter(
            x=[fecha_pred], y=[pred_final],
            mode='markers', name=f'Predicción: {pred_final:.1f} t',
            marker=dict(color='#f59e0b', size=14, symbol='diamond',
                        line=dict(width=2, color='white')),
            hovertemplate='Predicción: %{y:.1f} t<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[fecha_pred, fecha_pred],
            y=[max(0, pred_final - margen), pred_final + margen],
            mode='lines', line=dict(color='#f59e0b', width=3),
            showlegend=False, hoverinfo='skip'
        ))

        fig.update_layout(
            template="plotly_dark", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="Toneladas", hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Estacionalidad ---
        st.subheader("Patrón estacional de la pesca")
        st.caption("Promedio histórico de capturas por mes.")

        estacional = df.groupby('mes')['desembarques_t'].mean().reset_index()
        estacional['mes_nombre'] = estacional['mes'].apply(lambda m: MESES[int(m) - 1][:3])
        colores = ['#f59e0b' if int(m) == mes_pred else '#0ea5e9'
                   for m in estacional['mes']]

        fig_est = go.Figure()
        fig_est.add_trace(go.Bar(
            x=estacional['mes_nombre'], y=estacional['desembarques_t'],
            marker_color=colores,
            hovertemplate='%{x}: %{y:.1f} t<extra></extra>'
        ))
        fig_est.update_layout(
            template="plotly_dark", height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="", yaxis_title="Toneladas (media)", showlegend=False
        )
        st.plotly_chart(fig_est, use_container_width=True)

        # --- Explicación del modelo ---
        with st.expander("¿Cómo funciona este modelo?"):
            st.markdown("""
El modelo utiliza una **regresión Ridge** para estimar cuánto pulpo llegará a puerto.

Se alimenta de **13 variables** que se obtienen automáticamente:

- **Índice Niño 1+2**: temperatura del mar frente a la costa peruana.
  Se descarga en tiempo real de la NOAA.
- **Índice SOI**: presión atmosférica en el Pacífico. También de la NOAA.
- **Retardos climáticos** (1, 3, 6 y 12 meses atrás): el clima pasado
  influye en la disponibilidad futura de pulpo.
- **Desembarque del mes anterior**: el dato más determinante.
- **Estacionalidad** (seno y coseno del mes): la pesca tiene un ritmo anual.

**Rendimiento en test** (últimos 24 meses): R² = 0.831, MAE = 22.9 t, MAPE = 20.3%.

**¿Cómo se actualizan los datos?** Cada vez que abres la app, consulta los
archivos públicos de la NOAA (que se actualizan alrededor del día 10 de cada mes).
Si la NOAA no está accesible, la app usa los datos locales.
            """)

        with st.expander("Detalle de las variables usadas en esta predicción"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Variables climáticas**")
                detalle = pd.DataFrame({
                    'Variable': ['Niño 1+2', 'SOI',
                                 'Niño 1+2 (hace 1m)', 'SOI (hace 1m)',
                                 'Niño 1+2 (hace 3m)', 'SOI (hace 3m)',
                                 'Niño 1+2 (hace 6m)', 'SOI (hace 6m)',
                                 'Niño 1+2 (hace 12m)', 'SOI (hace 12m)'],
                    'Valor': [
                        f"{nino_val:.2f} °C", f"{soi_val:.1f}",
                        f"{lags['nino12_lag1']:.2f} °C", f"{lags['soi_lag1']:.1f}",
                        f"{lags['nino12_lag3']:.2f} °C", f"{lags['soi_lag3']:.1f}",
                        f"{lags['nino12_lag6']:.2f} °C", f"{lags['soi_lag6']:.1f}",
                        f"{lags['nino12_lag12']:.2f} °C", f"{lags['soi_lag12']:.1f}",
                    ]
                })
                st.dataframe(detalle, use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Estacionalidad y componente autorregresivo**")
                detalle2 = pd.DataFrame({
                    'Variable': ['Mes (seno)', 'Mes (coseno)', 'Desembarque mes anterior'],
                    'Valor': [
                        f"{ejemplo['mes_sin'].values[0]:.4f}",
                        f"{ejemplo['mes_cos'].values[0]:.4f}",
                        f"{lags['desembarque_lag1']:.2f} t"
                    ]
                })
                st.dataframe(detalle2, use_container_width=True, hide_index=True)

    # ─── PESTAÑA 2: POR DEPARTAMENTO ───
    with tab2:
        if df_dep is not None:
            st.subheader("Distribución de capturas por departamento")
            st.caption("¿Dónde se pesca más pulpo a lo largo de la costa peruana? Fuente: IMARPE.")

            dep_total = (
                df_dep.groupby('departamento')['desembarque_t']
                .sum().sort_values(ascending=False).reset_index()
            )
            fig_dep = px.bar(
                dep_total, x='desembarque_t', y='departamento', orientation='h',
                labels={'desembarque_t': 'Toneladas totales', 'departamento': ''},
                template='plotly_dark', color='desembarque_t',
                color_continuous_scale='Blues'
            )
            fig_dep.update_layout(
                height=400, margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False, coloraxis_showscale=False
            )
            st.plotly_chart(fig_dep, use_container_width=True)

            st.subheader("Evolución temporal por departamento")
            st.caption("Desembarques anuales de los 5 departamentos con más capturas.")
            top_deps = dep_total['departamento'].head(5).tolist()
            dep_anual = (
                df_dep[df_dep['departamento'].isin(top_deps)]
                .groupby(['anio', 'departamento'])['desembarque_t']
                .sum().reset_index()
            )
            fig_ev = px.line(
                dep_anual, x='anio', y='desembarque_t', color='departamento',
                labels={'desembarque_t': 'Toneladas', 'anio': 'Año', 'departamento': 'Departamento'},
                template='plotly_dark'
            )
            fig_ev.update_layout(
                height=400, margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info(
                "**Datos por departamento no disponibles.**\n\n"
                "Sube `datos_departamento.parquet` junto a los demás archivos para activar esta sección."
            )

    # ═══════════════════════════════════════
    #  PIE DE PÁGINA
    # ═══════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.82rem; padding: 0.5rem 0 1rem;'>"
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
