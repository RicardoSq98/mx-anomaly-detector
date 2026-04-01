import os
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, abs, when, avg as _avg, stddev as _std

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
