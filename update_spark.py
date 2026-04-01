import os
import requests
import pandas as pd
import json
import google.generativeai as genai
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, abs, when, avg as _avg, stddev as _std

# 1. INICIALIZAR SPARK
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("AutomatedSparkPipeline") \
    .getOrCreate()

# 2. CONFIGURACIÓN DE API (Banxico y Google IA)
TOKEN = os.getenv("BANXICO_TOKEN") 
SERIE = 'SF43718' # Tipo de cambio FIX
URL = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIE}/datos'

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# 3. INGESTA DE DATOS DESDE BANXICO
headers = {'Bmx-Token': TOKEN}
response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    observations = data['bmx']['series'][0]['datos']
    
    # Convertir a Spark via Pandas
    pdf = pd.DataFrame(observations)
    pdf['dato'] = pd.to_numeric(pdf['dato'])
    df_spark = spark.createDataFrame(pdf)
    
    # Limpieza de fechas y nombres
    df_spark = df_spark.withColumn("fecha", to_date(col("fecha"), "dd/MM/yyyy")) \
                        .withColumnRenamed("dato", "tipo_cambio")
else:
    raise Exception(f"Error al conectar con Banxico: {response.status_code}")

# 4. FEATURE ENGINEERING (Cálculo de variaciones)
windowSpec = Window.orderBy("fecha")
df_features = df_spark.withColumn("precio_anterior", F.lag("tipo_cambio", 1).over(windowSpec)) \
    .withColumn("cambio_diario", F.col("tipo_cambio") - F.col("precio_anterior"))

df_features = df_features.na.drop()

# 5. DETECCIÓN DE ANOMALÍAS (Z-Score)
ventana_global = Window.partitionBy(F.lit(1))
df_stats = df_features.withColumn("avg_historico", _avg("cambio_diario").over(ventana_global)) \
                       .withColumn("std_historico", _std("cambio_diario").over(ventana_global))

# Bajamos a 2 el Z-Score para asegurar que encuentre eventos y no salga vacío
df_final = df_stats.withColumn(
    "z_score_final", 
    (F.col("cambio_diario") - F.col("avg_historico")) / F.col("std_historico")
).withColumn(
    "es_anomalia", 
    when(abs(F.col("z_score_final")) > 2, 1).otherwise(0)
)

# 6. INTEGRACIÓN CON IA (GEMINI)
noticias_dict = {} # Diccionario inicializado vacío

# Extraemos las anomalías
anomalias_recientes = df_final.filter(F.col("es_anomalia") == 1) \
                              .sort(F.col("fecha").desc()) \
                              .limit(10).collect()

if len(anomalias_recientes) > 0:
    print(f"Se encontraron {len(anomalias_recientes)} anomalías. Consultando IA...")
    for row in anomalias_recientes:
        fecha_str = str(row['fecha'])
        noticias_dict[fecha_str] = obtener_explicacion_ia(fecha_str, row['tipo_cambio'])
else:
    print("No se encontraron anomalías con Z-Score > 2 hoy.")

# 7. GUARDADO FINAL (IMPORTANTE: FUERA DE CUALQUIER IF)
# Esto garantiza que los archivos existan para que GitHub Actions no falle

# Guardar Parquet
df_final.toPandas().to_parquet("datos_anomalias.parquet")

# Guardar JSON (aunque esté vacío {})
with open("noticias_contexto.json", "w") as f:
    json.dump(noticias_dict, f)

print("✅ Pipeline finalizado: noticias_contexto.json y datos_anomalias.parquet creados con éxito.")
