import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Monitor de Anomalías MX", layout="wide")

st.title("🇲🇽 Detector de Anomalías: Tipo de Cambio")
st.markdown("Este sistema monitorea en tiempo real el tipo de cambio USD/MXN utilizando la API de Banxico. No es una simple gráfica; es un ecosistema que detecta automáticamente crisis financieras o movimientos atípicos (anomalías) mediante estadística avanzada y procesamiento distribuido.")

# LEER PARQUET (Asegúrate de que el nombre coincida con el archivo que subas)
# Si el archivo está en tu repo, pon el nombre directo:
@st.cache_data
def load_data():
    # Buscamos el archivo .parquet que descargaste de Drive
    return pd.read_parquet("datos_anomalias.parquet")

df = load_data()
df['fecha'] = pd.to_datetime(df['fecha'])

# Título principal
st.title("🇲🇽 Monitor Inteligente: USD/MXN")

# Crear 3 columnas para el resumen rápido
col1, col2, col3 = st.columns(3)

# Supongamos que 'df' es tu DataFrame de Spark cargado
ultimo_precio = df.sort_values("fecha").iloc[-1]["tipo_cambio"]
precio_anterior = df.sort_values("fecha").iloc[-2]["tipo_cambio"]
delta = ultimo_precio - precio_anterior

with col1:
    st.metric("Precio Actual", f"${ultimo_precio:.2f}", f"{delta:.4f}")

with col2:
    total_anomalias = df[df["es_anomalia"] == 1].shape[0]
    st.metric("Anomalías Detectadas", total_anomalias)

with col3:
    # Fecha de la última actualización
    ultima_fecha = df["fecha"].max()
    st.metric("Última Actualización", str(ultima_fecha))

with st.expander("ℹ️ ¿Cómo funciona este monitor? (Resumen Técnico)"):
    st.write("""
    Este sistema utiliza **Apache Spark** para procesar el histórico del tipo de cambio de Banxico. 
    Aplica un modelo estadístico de **Z-Score** para identificar variaciones fuera de lo común.
    
    * **Anomalía (Puntos Rojos):** Ocurre cuando el cambio diario supera las **3 desviaciones estándar**.
    * **Automatización:** Los datos se actualizan solos cada mañana a las 8:00 AM mediante **GitHub Actions**.
    * **Tecnología:** PySpark, Parquet y Python 3.11.
    """)

# Gráfica Interactiva
fig = px.line(df, x='fecha', y='tipo_cambio', title='Histórico USD/MXN con Anomalías')
anomalias = df[df['es_anomalia'] == 1]
fig.add_scatter(x=anomalias['fecha'], y=anomalias['tipo_cambio'], 
                mode='markers', name='Anomalía', marker=dict(color='red', size=8))

st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Historial de Alertas Recientes")
df_anomalos = df[df["es_anomalia"] == 1].sort_values("fecha", ascending=False).head(5)

if not df_anomalos.empty:
    st.table(df_anomalos[["fecha", "tipo_cambio", "z_score_final"]])
else:
    st.write("No se han detectado anomalías en el periodo seleccionado.")
