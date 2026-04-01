import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Monitor de Anomalías MX - Spark",
    page_icon="🇲🇽",
    layout="wide"
)

# 2. Carga de datos con Cache para velocidad
@st.cache_data
def load_data():
    # Lee el archivo que genera tu robot de Spark
    df = pd.read_parquet("datos_anomalias.parquet")
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Ordenamos por fecha para que los cálculos de "último vs anterior" sean correctos
    return df.sort_values("fecha")

# Intentar cargar los datos
try:
    df = load_data()

    # 3. Encabezado Principal
    st.title("🇲🇽 Detector de Anomalías: Tipo de Cambio USD/MXN")
    st.markdown("""
    Este sistema monitorea en tiempo real el tipo de cambio utilizando la API de Banxico. 
    No es una simple gráfica; es un ecosistema que detecta automáticamente movimientos atípicos 
    mediante **Apache Spark** y estadística avanzada (**Z-Score**).
    """)

    # 4. Sección de Métricas (KPIs)
    st.divider()
    col1, col2, col3 = st.columns(3)

    # Cálculos para las métricas
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

    # 5. Resumen Técnico (Expander)
    with st.expander("ℹ️ ¿Cómo funciona la detección inteligente?"):
        st.write("""
        **Arquitectura del Pipeline:**
        * **Procesamiento:** El motor **Apache Spark** calcula la media y desviación estándar móvil.
        * **Z-Score:** Se marca una anomalía cuando el cambio diario se aleja más de **3 desviaciones estándar** ($$Z > 3$$).
        * **Automatización:** El proceso corre solo cada mañana mediante **GitHub Actions**.
        * **Almacenamiento:** Los datos se guardan en formato **Parquet** para máxima eficiencia.
        """)

    # 6. Gráfica Interactiva con Plotly
    st.subheader("📈 Análisis de Volatilidad Histórica")
    
    fig = px.line(df, x='fecha', y='tipo_cambio', 
                  title='Evolución del Peso Mexicano vs Dólar',
                  labels={'tipo_cambio': 'Precio (MXN)', 'fecha': 'Fecha'})
    
    # Agregar los puntos rojos de las anomalías
    anomalias = df[df['es_anomalia'] == 1]
    fig.add_scatter(x=anomalias['fecha'], y=anomalias['tipo_cambio'], 
                    mode='markers', name='Anomalía Crítica', 
                    marker=dict(color='red', size=9, symbol='x'))

    # Mejorar el diseño de la gráfica
    fig.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 7. Tabla de Alertas Recientes
    st.subheader("🚨 Historial de Alertas Recientes")
    df_alertas = df[df["es_anomalia"] == 1].sort_values("fecha", ascending=False).head(5)

    if not df_alertas.empty:
        # Formateamos la tabla para que se vea limpia
        st.dataframe(
            df_alertas[["fecha", "tipo_cambio", "z_score_final"]].style.format({
                "tipo_cambio": "{:.4f}",
                "z_score_final": "{:.2f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No se han detectado anomalías en el periodo actual.")

except Exception as e:
    st.error(f"Esperando datos del robot de Spark... (Error: {e})")
    st.info("Asegúrate de que el archivo 'datos_anomalias.parquet' exista en tu repositorio.")

# Pie de página
st.caption("Proyecto de Ingeniería de Datos | Apache Spark + GitHub Actions + Streamlit")
