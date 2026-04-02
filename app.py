import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# 1. Configuración de la página
st.set_page_config(
    page_title="Monitor de Anomalías MX - Spark",
    page_icon="🇲🇽",
    layout="wide"
)

# 2. Carga de datos con Cache
@st.cache_data
def load_data():
    df = pd.read_parquet("datos_anomalias.parquet")
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.sort_values("fecha")

# Intentar cargar los datos
try:
    df = load_data()

    # 3. Encabezado Principal
    st.title("🇲🇽 Detector de Anomalías: Tipo de Cambio USD/MXN")
    st.markdown("""
    Este sistema monitorea en tiempo real el tipo de cambio utilizando la API de Banxico. 
    Detecta automáticamente movimientos atípicos mediante **Apache Spark** y **Z-Score**.
    """)

    # 4. Sección de Métricas (KPIs)
    st.divider()
    col1, col2, col3 = st.columns(3)

    ultimo_registro = df.iloc[-1]
    penultimo_registro = df.iloc[-2]
    ultimo_precio = ultimo_registro["tipo_cambio"]
    delta_precio = ultimo_precio - penultimo_registro["tipo_cambio"]
    total_anomalias = df[df["es_anomalia"] == 1].shape[0]
    ultima_fecha = df["fecha"].max().strftime('%d/%m/%Y')

    with col1:
        st.metric("Precio Actual (FIX)", f"${ultimo_precio:.4f}", f"{delta_precio:.4f}")
    with col2:
        st.metric("Anomalías Detectadas", f"{total_anomalias} eventos", "Histórico")
    with col3:
        st.metric("Última Actualización", ultima_fecha, "Robot Activo ✅")

    # 5. Cargar contexto de la IA y mostrar Expander técnico
    noticias_contexto = {}
    if os.path.exists("noticias_contexto.json"):
        with open("noticias_contexto.json", "r") as f:
            noticias_contexto = json.load(f)

    with st.expander("ℹ️ ¿Cómo funciona la detección inteligente?"):
        st.write("""
        **Arquitectura del Pipeline:**
        * **Procesamiento:** El motor **Apache Spark** calcula la media y desviación estándar móvil.
        * **Z-Score:** Se marca anomalía si el cambio diario supera **3 desviaciones estándar**.
        * **IA Contextual:** Gemini analiza eventos macroeconómicos en las fechas de los picos.
        """)

    # 6. Gráfica Interactiva con Plotly
    st.subheader("📈 Análisis con IA de Anomalías USD/MXN")
    
    fig = px.line(df, x='fecha', y='tipo_cambio', 
                  labels={'tipo_cambio': 'Precio (MXN)', 'fecha': 'Fecha'})
    
    # Agregar los puntos rojos de las anomalías
    anomalias = df[df['es_anomalia'] == 1]
    fig.add_scatter(x=anomalias['fecha'], y=anomalias['tipo_cambio'], 
                    mode='markers', name='Anomalía', marker=dict(color='red', size=9))

    # --- AGREGAR EXPLICACIONES DE LA IA ---
    for fecha_str, explicacion in noticias_contexto.items():
        fecha_dt = pd.to_datetime(fecha_str)
        puntos_fecha = df[df['fecha'] == fecha_dt]
        
        if not puntos_fecha.empty:
            y_val = puntos_fecha['tipo_cambio'].iloc[0]
            fig.add_annotation(
                x=fecha_str, y=y_val,
                text=f"🤖 {explicacion}",
                showarrow=True, arrowhead=2, arrowcolor="#FF4B4B",
                ax=0, ay=-50, bgcolor="orange", bordercolor="#FF4B4B",
                borderwidth=1, borderpad=4, font=dict(size=12), opacity=0.9
            )

    fig.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 7. Tabla de Alertas Recientes
    st.subheader("🚨 Historial de Alertas Recientes")
    df_alertas = df[df["es_anomalia"] == 1].sort_values("fecha", ascending=False).head(5)

    if not df_alertas.empty:
        st.dataframe(
            df_alertas[["fecha", "tipo_cambio", "z_score_final"]].style.format({
                "tipo_cambio": "{:.4f}",
                "z_score_final": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No se han detectado anomalías en el periodo actual.")

except Exception as e:
    st.error(f"Error al cargar la App: {e}")

st.caption("Proyecto de Ingeniería de Datos | Apache Spark + Gemini IA + GitHub Actions")
