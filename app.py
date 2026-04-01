import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import requests
from datetime import datetime

# ─── Configuración ───
st.set_page_config(
    page_title="TFM · Predicción Pulpo Perú — María Ojeda García",
    page_icon="icon_pulpo.png",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem 1.4rem; border-radius: 12px;
        border-left: 4px solid #0ea5e9; margin-bottom: 0.8rem;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.78rem; margin: 0 0 0.2rem 0;
        text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .metric-card .detail { color: #64748b; font-size: 0.78rem; margin-top: 0.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  DESCARGA AUTOMÁTICA NOAA
# ═══════════════════════════════════════════

MESES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
         "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

@st.cache_data(ttl=86400)
def fetch_nino12_noaa():
    """Descarga Niño 1+2 mensual (anomalía) del archivo sstoi.indices de la NOAA.
    Formato: YR MON NINO1+2_abs ANOM NINO3_abs ANOM NINO34_abs ANOM NINO4_abs ANOM"""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = []
        for line in r.text.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    data.append({
                        'anio': int(parts[0]), 'mes': int(parts[1]),
                        'nino12_anom': float(parts[3]),  # columna 4 = anomalía
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(data) if data else None
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_soi_noaa():
    """Descarga SOI mensual del archivo soi de la NOAA.
    Formato ancho: YEAR JAN FEB ... DEC (con cabecera de 3 líneas)."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = []
        for line in r.text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(float(parts[0]))
                    if year < 1950 or year > 2100:
                        continue
                    for m in range(1, 13):
                        val = float(parts[m])
                        if val < -90:
                            continue
                        data.append({'anio': year, 'mes': m, 'soi': val})
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(data) if data else None
    except Exception:
        return None


def get_climate(anio, mes, df_local, nino_noaa, soi_noaa):
    """Busca nino12_anom y soi. Prioridad: NOAA > dataset local."""
    nino, soi, fuente = None, None, "no disponible"

    if nino_noaa is not None:
        m = nino_noaa[(nino_noaa['anio'] == anio) & (nino_noaa['mes'] == mes)]
        if len(m) > 0:
            nino = float(m.iloc[0]['nino12_anom'])
            fuente = "NOAA (en línea)"
    if soi_noaa is not None:
        m = soi_noaa[(soi_noaa['anio'] == anio) & (soi_noaa['mes'] == mes)]
        if len(m) > 0:
            soi = float(m.iloc[0]['soi'])

    if nino is None:
        m = df_local[(df_local['anio'] == anio) & (df_local['mes'] == mes)]
        if len(m) > 0:
            nino = float(m.iloc[0]['nino12_anom'])
            soi = float(m.iloc[0]['soi'])
            fuente = "dataset local"

    return nino, soi, fuente


# ═══════════════════════════════════════════
#  CARGA DE ARCHIVOS
# ═══════════════════════════════════════════

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

st.title("Predicción de Desembarques de Pulpo — Perú")

st.markdown(
    "**Trabajo Fin de Máster** · María Ojeda García  \n"
    "Estimación del pulpo que llegará a los puertos peruanos "
    "a partir de los índices climáticos del Pacífico (Niño 1+2 y SOI) "
    "y del historial de capturas. "
    "Los datos climáticos se descargan automáticamente de la NOAA."
)

# ═══════════════════════════════════════════
#  APP PRINCIPAL
# ═══════════════════════════════════════════

try:
    model, scaler, feature_names, df, df_dep = load_assets()

    # Fechas para gráficas
    df_plot = df.copy()
    df_plot['fecha'] = pd.to_datetime(
        df_plot['anio'].astype(int).astype(str) + '-' +
        df_plot['mes'].astype(int).astype(str).str.zfill(2) + '-01'
    )

    ultimo = df.iloc[-1]
    ultimo_anio = int(ultimo['anio'])
    ultimo_mes = int(ultimo['mes'])

    # Descargar NOAA
    with st.spinner("Consultando datos climáticos de la NOAA..."):
        nino_noaa = fetch_nino12_noaa()
        soi_noaa = fetch_soi_noaa()
    noaa_ok = nino_noaa is not None and soi_noaa is not None

    # ═══════════════════════════════════════
    #  BARRA LATERAL
    # ═══════════════════════════════════════

    st.sidebar.header("Selecciona un mes")

    if noaa_ok:
        st.sidebar.success("Conectado a la NOAA")
        ultimo_noaa_anio = int(nino_noaa['anio'].max())
        ultimo_noaa_mes = int(nino_noaa[nino_noaa['anio'] == ultimo_noaa_anio]['mes'].max())
        st.sidebar.caption(f"Último dato NOAA: {MESES[ultimo_noaa_mes - 1]} {ultimo_noaa_anio}")
        max_anio = ultimo_noaa_anio + 1
    else:
        st.sidebar.warning("Sin conexión a NOAA. Usando datos locales.", icon="⚠️")
        st.sidebar.caption(f"Último dato local: {MESES[ultimo_mes - 1]} {ultimo_anio}")
        max_anio = ultimo_anio + 1

# Fecha por defecto: año actual y mes anterior
    ahora = datetime.now()
    mes_anterior = (ahora.month - 2) % 12 + 1
    anio_mes_anterior = ahora.year if ahora.month > 1 else ahora.year - 1
 
    años_lista = list(range(max_anio, 1996, -1))
    try:
        idx_anio = años_lista.index(anio_mes_anterior)
    except ValueError:
        idx_anio = 0
 
    anio_pred = st.sidebar.selectbox("Año", años_lista, index=idx_anio)
    mes_pred = st.sidebar.selectbox("Mes", list(range(1, 13)),
                                     format_func=lambda m: MESES[m - 1],
                                     index=mes_anterior - 1)

    # --- Buscar clima ---
    nino_val, soi_val, fuente_clima = get_climate(anio_pred, mes_pred, df, nino_noaa, soi_noaa)

    datos_estimados = False
    if nino_val is None:
        if noaa_ok:
            last = nino_noaa.iloc[-1]
            nino_val = float(last['nino12_anom'])
            soi_m = soi_noaa[(soi_noaa['anio'] == int(last['anio'])) &
                             (soi_noaa['mes'] == int(last['mes']))]
            soi_val = float(soi_m.iloc[0]['soi']) if len(soi_m) > 0 else float(ultimo['soi'])
            fuente_clima = f"Último NOAA ({MESES[int(last['mes']) - 1]} {int(last['anio'])})"
        else:
            nino_val = float(ultimo['nino12_anom'])
            soi_val = float(ultimo['soi'])
            fuente_clima = f"Último local ({MESES[ultimo_mes - 1]} {ultimo_anio})"
        datos_estimados = True

    if soi_val is None:
        soi_val = float(ultimo['soi'])

    st.sidebar.divider()
    st.sidebar.markdown("**Datos climáticos obtenidos:**")
    st.sidebar.metric("Niño 1+2 (anomalía)", f"{nino_val:+.2f} °C")
    st.sidebar.metric("SOI", f"{soi_val:.1f}")
    st.sidebar.caption(f"Fuente: {fuente_clima}")

    if datos_estimados:
        st.sidebar.warning(
            f"No hay datos para {MESES[mes_pred - 1]} {anio_pred}. "
            f"Se usa el último valor disponible.", icon="⚠️")

    # --- Lags y desembarque anterior ---
    fila = df[(df['anio'] == anio_pred) & (df['mes'] == mes_pred)]

    if len(fila) > 0:
        row = fila.iloc[0]
        lags = {k: row[k] for k in feature_names if k.startswith(('nino12_lag','soi_lag','desembarque_lag'))}
        desemb_real = float(row['desembarque_t'])
        hay_real = True
    else:
        lags = {
            'nino12_lag1': ultimo['nino12_anom'], 'soi_lag1': ultimo['soi'],
            'nino12_lag3': ultimo['nino12_lag1'], 'soi_lag3': ultimo['soi_lag1'],
            'nino12_lag6': ultimo['nino12_lag3'], 'soi_lag6': ultimo['soi_lag3'],
            'nino12_lag12': ultimo['nino12_lag6'], 'soi_lag12': ultimo['soi_lag6'],
            'desembarque_lag1': ultimo['desembarque_t'],
        }
        desemb_real = None
        hay_real = False

    # --- Predicción ---
    ejemplo = pd.DataFrame([{
        'nino12_anom': nino_val, 'soi': soi_val,
        'mes_sin': np.sin(2 * np.pi * mes_pred / 12),
        'mes_cos': np.cos(2 * np.pi * mes_pred / 12),
        **lags,
    }])
    ejemplo = ejemplo[feature_names]
    pred_final = max(0, model.predict(scaler.transform(ejemplo))[0])
    margen = pred_final * 0.227  # MAPE 22.7%
# La imagen se estirará hasta tocar los bordes de la barra lateral y transparente
    st.sidebar.markdown(
    """
    <style>
        /* Seleccionamos las imágenes dentro de la barra lateral */
        [data-testid="stSidebar"] img {
            opacity: 0.4; 
            filter: saturate(70%); 
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.image("mapa_pulpo.png", use_container_width=True)
    # ═══════════════════════════════════════
    #  PESTAÑAS
    # ═══════════════════════════════════════

    tab1, tab2, tab3 = st.tabs(["Vista general", "Por departamento", "Rendimiento del modelo"])

    with tab1:

        # --- Tarjetas ---
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if hay_real:
                st.markdown(f"""<div class="metric-card">
                    <h3>Dato real registrado</h3>
                    <p class="value">{desemb_real:.1f} t</p>
                    <p class="detail">{MESES[mes_pred-1]} {anio_pred}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="metric-card">
                    <h3>Último dato registrado</h3>
                    <p class="value">{float(ultimo['desembarque_t']):.1f} t</p>
                    <p class="detail">{MESES[ultimo_mes-1]} {ultimo_anio}</p>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div class="metric-card" style="border-left-color: #f59e0b;">
                <h3>Predicción del modelo</h3>
                <p class="value">{pred_final:.1f} t</p>
                <p class="detail">{MESES[mes_pred-1]} {anio_pred} · rango: {max(0,pred_final-margen):.0f}–{pred_final+margen:.0f} t</p>
            </div>""", unsafe_allow_html=True)

        with c3:
            if hay_real:
                err = abs(desemb_real - pred_final)
                err_pct = (err / desemb_real * 100) if desemb_real > 0 else 0
                col_err = "#22c55e" if err_pct < 20 else "#f59e0b" if err_pct < 35 else "#ef4444"
                st.markdown(f"""<div class="metric-card" style="border-left-color: {col_err};">
                    <h3>Error de la predicción</h3>
                    <p class="value" style="color: {col_err};">{err_pct:.1f}%</p>
                    <p class="detail">{err:.1f} t de diferencia</p>
                </div>""", unsafe_allow_html=True)
            else:
                media = df['desembarque_t'].mean()
                diff = ((pred_final - media) / media) * 100
                s = "+" if diff > 0 else ""
                col_d = "#22c55e" if diff > 0 else "#ef4444"
                st.markdown(f"""<div class="metric-card" style="border-left-color: {col_d};">
                    <h3>vs. media histórica</h3>
                    <p class="value" style="color: {col_d};">{s}{diff:.1f}%</p>
                    <p class="detail">Media: {media:.1f} t/mes</p>
                </div>""", unsafe_allow_html=True)

        with c4:
            if nino_val > 1.5:
                estado, col_cl = "El Niño (cálido)", "#ef4444"
            elif nino_val < -0.5:
                estado, col_cl = "La Niña (frío)", "#3b82f6"
            else:
                estado, col_cl = "Neutro", "#a3a3a3"
            st.markdown(f"""<div class="metric-card" style="border-left-color: {col_cl};">
                <h3>Condición climática</h3>
                <p class="value" style="font-size: 1.3rem; color: {col_cl};">{estado}</p>
                <p class="detail">Niño 1+2 anom: {nino_val:+.2f} °C · SOI: {soi_val:.1f}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # --- Gráfica histórica ---
        st.subheader("Evolución de las capturas mensuales")
        if hay_real:
            st.caption("La línea azul son los desembarques reales. El diamante naranja es lo que el modelo habría predicho.")
        else:
            st.caption("La línea azul son los desembarques reales. El diamante naranja es la predicción del modelo.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot['fecha'], y=df_plot['desembarque_t'],
            mode='lines', name='Desembarques reales',
            line=dict(color='#0ea5e9', width=1.5),
            fill='tozeroy', fillcolor='rgba(14,165,233,0.07)',
            hovertemplate='%{x|%b %Y}: %{y:.1f} t<extra></extra>'
        ))

        fecha_pred = pd.Timestamp(year=anio_pred, month=mes_pred, day=1)
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
            template="plotly_dark", height=450, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="Toneladas"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Estacionalidad ---
        st.subheader("Patrón estacional de la pesca")
        st.caption("Promedio histórico de capturas por mes. El mes seleccionado aparece resaltado.")

        est = df.groupby('mes')['desembarque_t'].mean().reset_index()
        est['lbl'] = est['mes'].apply(lambda m: MESES[int(m)-1][:3])
        colores = ['#f59e0b' if int(m) == mes_pred else '#0ea5e9' for m in est['mes']]

        fig_e = go.Figure(go.Bar(
            x=est['lbl'], y=est['desembarque_t'], marker_color=colores,
            hovertemplate='%{x}: %{y:.1f} t<extra></extra>'
        ))
        fig_e.update_layout(template="plotly_dark", height=300, showlegend=False,
                            margin=dict(l=10,r=10,t=20,b=10),
                            xaxis_title="", yaxis_title="Toneladas (media)")
        st.plotly_chart(fig_e, use_container_width=True)

        # --- Explicación ---
        with st.expander("¿Cómo funciona este modelo?"):
            st.markdown("""
El modelo usa una **regresión Ridge** (α=10) para estimar cuánto pulpo llegará a puerto.

Se alimenta de **13 variables** obtenidas automáticamente:

- **Anomalía Niño 1+2**: cuánto se desvía la temperatura del mar frente a Perú
  respecto a lo normal. Se descarga de la NOAA en tiempo real.
- **SOI**: presión atmosférica en el Pacífico. También de la NOAA.
- **Retardos climáticos** (1, 3, 6 y 12 meses): el clima pasado influye en la
  disponibilidad futura de pulpo.
- **Desembarque del mes anterior**: el dato más determinante.
- **Estacionalidad** (seno y coseno del mes).

**Rendimiento en test** (últimos 24 meses):
R² = 0.831 · MAE = 22.9 t · MAPE = 22.7%

Los datos climáticos se actualizan automáticamente cada vez que se abre la app
(la NOAA los publica alrededor del día 10 de cada mes).
            """)

        with st.expander("Detalle de las variables usadas"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Variables climáticas**")
                st.dataframe(pd.DataFrame({
                    'Variable': ['Niño 1+2 (anomalía)', 'SOI',
                                 'Niño 1+2 lag 1m', 'SOI lag 1m',
                                 'Niño 1+2 lag 3m', 'SOI lag 3m',
                                 'Niño 1+2 lag 6m', 'SOI lag 6m',
                                 'Niño 1+2 lag 12m', 'SOI lag 12m'],
                    'Valor': [f"{nino_val:+.2f} °C", f"{soi_val:.1f}",
                              f"{lags['nino12_lag1']:+.2f} °C", f"{lags['soi_lag1']:.1f}",
                              f"{lags['nino12_lag3']:+.2f} °C", f"{lags['soi_lag3']:.1f}",
                              f"{lags['nino12_lag6']:+.2f} °C", f"{lags['soi_lag6']:.1f}",
                              f"{lags['nino12_lag12']:+.2f} °C", f"{lags['soi_lag12']:.1f}"]
                }), use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Estacionalidad y autorregresivo**")
                st.dataframe(pd.DataFrame({
                    'Variable': ['Mes (seno)', 'Mes (coseno)', 'Desembarque mes anterior'],
                    'Valor': [f"{ejemplo['mes_sin'].values[0]:.4f}",
                              f"{ejemplo['mes_cos'].values[0]:.4f}",
                              f"{lags['desembarque_lag1']:.2f} t"]
                }), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════
    #  PESTAÑA 2: POR DEPARTAMENTO
    # ═══════════════════════════════════════

    with tab2:
        if df_dep is not None:
            st.subheader("Distribución de capturas por departamento")
            st.caption("¿Dónde se pesca más pulpo a lo largo de la costa peruana? Fuente: IMARPE.")

            dep_tot = (df_dep.groupby('departamento')['desembarque_t']
                       .sum().sort_values(ascending=False).reset_index())
            fig_d = px.bar(dep_tot, x='desembarque_t', y='departamento', orientation='h',
                           labels={'desembarque_t':'Toneladas totales','departamento':''},
                           template='plotly_dark', color='desembarque_t',
                           color_continuous_scale='Blues')
            fig_d.update_layout(height=400, margin=dict(l=10,r=10,t=20,b=10),
                                showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)

            st.subheader("Evolución temporal por departamento")
            st.caption("Desembarques anuales de los 5 departamentos con más capturas.")
            top5 = dep_tot['departamento'].head(5).tolist()
            dep_a = (df_dep[df_dep['departamento'].isin(top5)]
                     .groupby(['anio','departamento'])['desembarque_t']
                     .sum().reset_index())
            fig_ev = px.line(dep_a, x='anio', y='desembarque_t', color='departamento',
                             labels={'desembarque_t':'Toneladas','anio':'Año','departamento':'Departamento'},
                             template='plotly_dark')
            fig_ev.update_layout(height=400, margin=dict(l=10,r=10,t=20,b=10),
                                 legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("**Datos por departamento no disponibles.**\n\n"
                    "Sube `datos_departamento.parquet` junto a los demás archivos para activar esta sección.")

    # ═══════════════════════════════════════
    #  PESTAÑA 3: RENDIMIENTO DEL MODELO
    # ═══════════════════════════════════════

    with tab3:
        st.subheader("¿Cómo de bien predice el modelo?")
        st.caption(
            "El modelo se evaluó reservando los **últimos 24 meses** de datos (enero 2024 – diciembre 2025) "
            "como conjunto de test: datos que el modelo nunca vio durante el entrenamiento."
        )

        from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

        n_test = 24
        X_all = df[feature_names].values
        y_all = df['desembarque_t'].values
        X_test_eval = X_all[-n_test:]
        y_test_eval = y_all[-n_test:]

        X_test_sc_eval = scaler.transform(X_test_eval)
        preds_eval = model.predict(X_test_sc_eval)
        preds_eval_clip = np.maximum(0, preds_eval)

        r2 = r2_score(y_test_eval, preds_eval_clip)
        mae = mean_absolute_error(y_test_eval, preds_eval_clip)
        mape = mean_absolute_percentage_error(y_test_eval, preds_eval_clip) * 100

        # --- Métricas en tarjetas ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card" style="border-left-color: #22c55e;">
                <h3>R² (coeficiente de determinación)</h3>
                <p class="value">{r2:.3f}</p>
                <p class="detail">El modelo explica el {r2*100:.0f}% de la variabilidad</p>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card" style="border-left-color: #f59e0b;">
                <h3>MAE (error medio absoluto)</h3>
                <p class="value">{mae:.1f} t</p>
                <p class="detail">De media, se desvía {mae:.1f} toneladas del valor real</p>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card" style="border-left-color: #3b82f6;">
                <h3>MAPE (error porcentual medio)</h3>
                <p class="value">{mape:.1f}%</p>
                <p class="detail">Las predicciones se desvían de media un {mape:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # --- Gráfica: Predicción vs Real ---
        st.subheader("Predicción vs realidad (últimos 24 meses)")
        st.caption(
            "La línea negra son los desembarques que realmente ocurrieron. "
            "La línea azul es lo que el modelo habría predicho. "
            "La zona sombreada es la diferencia (el error)."
        )

        fechas_test = df[['anio', 'mes']].iloc[-n_test:]
        etiquetas = [f"{int(r['anio'])}-{int(r['mes']):02d}" for _, r in fechas_test.iterrows()]

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=etiquetas, y=y_test_eval,
            mode='lines+markers', name='Real',
            line=dict(color='white', width=2), marker=dict(size=5),
            hovertemplate='%{x}: %{y:.1f} t<extra>Real</extra>'
        ))
        fig_perf.add_trace(go.Scatter(
            x=etiquetas, y=preds_eval_clip,
            mode='lines+markers', name='Predicción Ridge',
            line=dict(color='#0ea5e9', width=2, dash='dash'),
            marker=dict(size=5, symbol='square'),
            hovertemplate='%{x}: %{y:.1f} t<extra>Predicción</extra>'
        ))
        fig_perf.add_trace(go.Scatter(
            x=etiquetas + etiquetas[::-1],
            y=list(y_test_eval) + list(preds_eval_clip[::-1]),
            fill='toself', fillcolor='rgba(14,165,233,0.12)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig_perf.update_layout(
            template="plotly_dark", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="Toneladas", xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # --- Gráfica: Error por mes ---
        st.subheader("Error por mes")
        st.caption(
            "Cada barra muestra cuánto se equivocó el modelo en ese mes. "
            "Azul = subestimó (predijo menos de lo real). Rojo = sobreestimó."
        )

        errores = y_test_eval - preds_eval_clip
        colores_err = ['#0ea5e9' if e >= 0 else '#ef4444' for e in errores]

        fig_err = go.Figure(go.Bar(
            x=etiquetas, y=errores, marker_color=colores_err,
            hovertemplate='%{x}: %{y:+.1f} t<extra></extra>'
        ))
        fig_err.add_hline(y=0, line_color='white', line_width=0.5)
        fig_err.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="", yaxis_title="Error (t)",
            xaxis=dict(tickangle=-45), showlegend=False
        )
        st.plotly_chart(fig_err, use_container_width=True)

        # --- Comparativa con baselines ---
        st.subheader("¿Supera el modelo a las predicciones más simples?")
        st.caption(
            "Para que un modelo tenga valor, debe superar a las estrategias triviales. "
            "Aquí se compara con repetir el mes anterior (persistencia) y con repetir el mismo mes del año pasado."
        )

        baseline_lag1 = df['desembarque_lag1'].values[-n_test:]
        r2_naive = r2_score(y_test_eval, baseline_lag1)
        mae_naive = mean_absolute_error(y_test_eval, baseline_lag1)

        baseline_lag12 = y_all[-n_test - 12:-12]
        if len(baseline_lag12) == n_test:
            r2_estac = r2_score(y_test_eval, baseline_lag12)
            mae_estac = mean_absolute_error(y_test_eval, baseline_lag12)
        else:
            r2_estac, mae_estac = None, None

        comparativa = pd.DataFrame([
            {"Método": "Ridge (nuestro modelo)", "R²": f"{r2:.3f}", "MAE (t)": f"{mae:.1f}", "": "✅ Ganador"},
            {"Método": "Persistencia (repetir mes anterior)", "R²": f"{r2_naive:.3f}", "MAE (t)": f"{mae_naive:.1f}", "": ""},
        ])
        if r2_estac is not None:
            comparativa = pd.concat([comparativa, pd.DataFrame([
                {"Método": "Estacional (mismo mes, año anterior)", "R²": f"{r2_estac:.3f}", "MAE (t)": f"{mae_estac:.1f}", "": ""}
            ])], ignore_index=True)

        st.dataframe(comparativa, use_container_width=True, hide_index=True)

        mejora = mae_naive - mae
        st.markdown(
            f"El modelo Ridge reduce el error en **{mejora:.1f} toneladas por mes** "
            f"respecto a simplemente repetir el dato del mes anterior."
        )

    # ═══════════════════════════════════════
    #  PIE DE PÁGINA
    # ═══════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.82rem; padding:0.5rem 0 1rem;'>"
        "TFM — Predicción de desembarques de pulpo en Perú mediante variables climáticas<br>"
        "María Ojeda García · Máster en Big Data · 2025<br>"
        "Fuentes: IMARPE · NOAA (CPC) · BOM Australia</div>",
        unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"Archivo no encontrado: {e}")
    st.info("Archivos necesarios: `modelo_ridge_final.pkl`, `scaler_ridge.pkl`, "
            "`feature_names.pkl`, `datos_modelo.parquet`, `icon_pulpo.png`, "
            "y (opcional) `datos_departamento.parquet`.")
except Exception as e:
    st.error(f"Error inesperado: {e}")
