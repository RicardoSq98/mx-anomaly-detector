import os
import requests
import pandas as pd
import json
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

# 3. LA FUNCIÓN (Cópiala tal cual para que no falte)
def obtener_explicacion_ia(fecha, valor):
    token = os.getenv("HF_TOKEN")
    API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {token}"}

    prompt = f"<s>[INST] Eres un analista financiero. El {fecha} el dólar llegó a {valor} MXN. Responde en 8 palabras en español por qué hubo esa anomalía. [/INST]"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        res_json = response.json()
        
        # Hugging Face devuelve una lista con el texto
        if isinstance(res_json, list) and 'generated_text' in res_json[0]:
            texto = res_json[0]['generated_text'].strip()
            return texto
        else:
            print(f"Error HF: {res_json}")
            return "Ajuste técnico por volatilidad global."
            
    except Exception as err:
        print(f"Error de conexión HF: {err}")
        return "Movimiento de mercado estándar."

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
