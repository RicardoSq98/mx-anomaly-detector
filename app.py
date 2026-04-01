import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Monitor de Anomalías MX", layout="wide")

st.title("🇲🇽 Detector de Anomalías: Tipo de Cambio")
st.markdown("Sistema profesional procesado con **Apache Spark** y almacenado en **Parquet**.")

# LEER PARQUET (Asegúrate de que el nombre coincida con el archivo que subas)
# Si el archivo está en tu repo, pon el nombre directo:
@st.cache_data
def load_data():
    # Buscamos el archivo .parquet que descargaste de Drive
    return pd.read_parquet("part-00000-5a19e0e5-5f4e-4e1d-8b85-e730434f9cd0-c000.snappy.parquet")

df = load_data()
df['fecha'] = pd.to_datetime(df['fecha'])

# Gráfica Interactiva
fig = px.line(df, x='fecha', y='tipo_cambio', title='Histórico USD/MXN con Anomalías')
anomalias = df[df['es_anomalia'] == 1]
fig.add_scatter(x=anomalias['fecha'], y=anomalias['tipo_cambio'], 
                mode='markers', name='Anomalía', marker=dict(color='red', size=8))

st.plotly_chart(fig, use_container_width=True)
