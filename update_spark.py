import os
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, abs, when, avg as _avg, stddev as _std
import google.generativeai as genai
import json

# 1. INICIALIZAR SPARK (Configuración para GitHub Actions)
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("AutomatedSparkPipeline") \
    .getOrCreate()

# 2. CONFIGURACIÓN DE API (Usando el Secret de GitHub)
TOKEN = os.getenv("BANXICO_TOKEN") 
SERIE = 'SF43718' 
URL = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIE}/datos'

# 3. INGESTA DE DATOS
headers = {'Bmx-Token': TOKEN}
response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    observations = data['bmx']['series'][0]['datos']
    
    # Convertir a Spark
    pdf = pd.DataFrame(observations)
    pdf['dato'] = pd.to_numeric(pdf['dato'])
    df_spark = spark.createDataFrame(pdf)
    
    # Limpieza
    df_spark = df_spark.withColumn("fecha", to_date(col("fecha"), "dd/MM/yyyy")) \
                       .withColumnRenamed("dato", "tipo_cambio")
else:
    raise Exception(f"Error al conectar con Banxico: {response.status_code}")

# 4. FEATURE ENGINEERING (Ventana Móvil)
windowSpec = Window.orderBy("fecha")
df_features = df_spark.withColumn("precio_anterior", F.lag("tipo_cambio", 1).over(windowSpec)) \
    .withColumn("cambio_diario", F.col("tipo_cambio") - F.col("precio_anterior")) \
    .withColumn("media_movil_7d", F.avg("tipo_cambio").over(windowSpec.rowsBetween(-6, 0))) \
    .withColumn("desv_estandar_7d", F.stddev("tipo_cambio").over(windowSpec.rowsBetween(-6, 0)))

df_features = df_features.na.drop()

# 5. DETECCIÓN DE ANOMALÍAS (Z-Score)
ventana_global = Window.partitionBy()
df_stats = df_features.withColumn("avg_historico", _avg("cambio_diario").over(ventana_global)) \
                       .withColumn("std_historico", _std("cambio_diario").over(ventana_global))

df_final = df_stats.withColumn(
    "z_score_final", 
    (F.col("cambio_diario") - F.col("avg_historico")) / F.col("std_historico")
).withColumn(
    "es_anomalia", 
    when(abs(F.col("z_score_final")) > 3, 1).otherwise(0)
)

# 6. GUARDADO FINAL (Para GitHub y Streamlit)
# En lugar de Drive, guardamos un solo archivo Parquet en la carpeta del proyecto
# Usamos .coalesce(1) para que Spark no genere 200 archivos pequeños
df_final.toPandas().to_parquet("datos_anomalias.parquet")

print("Pipeline ejecutado con éxito. Archivo datos_anomalias.parquet actualizado.")

# Configurar la IA
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def obtener_explicacion_ia(fecha, valor):
    prompt = f"""
    Actúa como un analista financiero Senior. 
    El tipo de cambio USD/MXN tuvo una anomalía el día {fecha} llegando a {valor}.
    Busca en tu base de datos qué eventos macroeconómicos o políticos ocurrieron en esa fecha 
    y dame un resumen de máximo 10 palabras de por qué se disparó el dólar. 
    Sé muy directo. Ejemplo: 'Incertidumbre por elecciones y alza en tasas de la FED.'
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        print(f"Error en Gemini para {fecha}: {e}")
            return "Ajuste técnico de mercado."

# Si detectamos anomalías hoy, generamos el JSON de noticias
anomalias_recientes = df_final.filter(F.col("es_anomalia") == 1).sort(F.col("fecha").desc()).limit(5).collect()

noticias_dict = {}
for row in anomalias_recientes:
    fecha_str = row['fecha'].strftime('%Y-%m-%d')
    noticias_dict[fecha_str] = obtener_explicacion_ia(fecha_str, row['tipo_cambio'])

# Guardar para que Streamlit lo lea
with open("noticias_contexto.json", "w") as f:
    json.dump(noticias_dict, f)
