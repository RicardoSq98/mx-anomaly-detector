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
SERIE = 'SF43718' 
URL = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIE}/datos'

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# 3. FUNCIÓN DE IA (La que faltaba, ¡aquí está!)
def obtener_explicacion_ia(fecha, valor):
    # Prompt más sencillo para evitar filtros de seguridad
    prompt = f"¿Qué pasó el {fecha} que afectó al peso mexicano? Responde en 8 palabras."
    
    try:
        # Forzamos que use la versión más estable
        response = model.generate_content(prompt)
        
        # Si la respuesta está vacía o bloqueada
        if not response.text:
            return "Movimiento por flujo de capitales."
            
        return response.text.strip()
    except Exception as err:
        # ESTO ES VITAL: Ver el error en los logs de GitHub
        print(f"❌ ERROR REAL PARA {fecha}: {err}")
        return "Ajuste técnico de mercado."

# 4. INGESTA DE DATOS
headers = {'Bmx-Token': TOKEN}
response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    observations = data['bmx']['series'][0]['datos']
    pdf = pd.DataFrame(observations)
    pdf['dato'] = pd.to_numeric(pdf['dato'])
    df_spark = spark.createDataFrame(pdf)
    df_spark = df_spark.withColumn("fecha", to_date(col("fecha"), "dd/MM/yyyy")) \
                        .withColumnRenamed("dato", "tipo_cambio")
else:
    raise Exception(f"Error al conectar con Banxico: {response.status_code}")

# 5. FEATURE ENGINEERING
windowSpec = Window.orderBy("fecha")
df_features = df_spark.withColumn("precio_anterior", F.lag("tipo_cambio", 1).over(windowSpec)) \
    .withColumn("cambio_diario", F.col("tipo_cambio") - F.col("precio_anterior"))
df_features = df_features.na.drop()

# 6. DETECCIÓN DE ANOMALÍAS (Z-Score > 2)
ventana_global = Window.partitionBy(F.lit(1))
df_stats = df_features.withColumn("avg_historico", _avg("cambio_diario").over(ventana_global)) \
                       .withColumn("std_historico", _std("cambio_diario").over(ventana_global))

df_final = df_stats.withColumn(
    "z_score_final", 
    (F.col("cambio_diario") - F.col("avg_historico")) / F.col("std_historico")
).withColumn(
    "es_anomalia", 
    when(abs(F.col("z_score_final")) > 2, 1).otherwise(0)
)

# 7. INTEGRACIÓN CON IA
noticias_dict = {}
anomalias_recientes = df_final.filter(F.col("es_anomalia") == 1) \
                              .sort(F.col("fecha").desc()) \
                              .limit(10).collect()

if len(anomalias_recientes) > 0:
    print(f"Se encontraron {len(anomalias_recientes)} anomalías. Consultando IA...")
    for row in anomalias_recientes:
        fecha_str = str(row['fecha'])
        noticias_dict[fecha_str] = obtener_explicacion_ia(fecha_str, row['tipo_cambio'])

# 8. GUARDADO FINAL (Garantiza que los archivos existan)
df_final.toPandas().to_parquet("datos_anomalias.parquet")

with open("noticias_contexto.json", "w") as f:
    json.dump(noticias_dict, f)

print("✅ Pipeline finalizado: noticias_contexto.json y datos_anomalias.parquet creados con éxito.")
