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

# 2. CONFIGURACIÓN DE API
TOKEN = os.getenv("BANXICO_TOKEN") 
SERIE = 'SF43718' 
URL = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIE}/datos'

# 3. FUNCIÓN DE IA MEJORADA (Prompt de Ingeniería)
def obtener_explicacion_ia(fecha, valor):
    token = os.getenv("HF_TOKEN")
    # Usamos el modelo Llama 3.2-1B Instruct para mejores respuestas
    API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.2-1B-Instruct"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Prompt formateado para que la IA entienda que es un analista financiero
    prompt_f = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Analiza brevemente: ¿Qué evento económico o político causó volatilidad en el peso mexicano el {fecha}? "
        f"Responde en una sola oración corta y profesional.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "inputs": prompt_f,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "stop": ["<|eot_id|>"]
        },
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return "Variación por volatilidad del mercado cambiario."

        res_json = response.json()
        
        if isinstance(res_json, list) and 'generated_text' in res_json[0]:
            # Limpiamos el texto generado para quitar el prompt
            full_text = res_json[0]['generated_text']
            # Extraemos solo lo que respondió el asistente después del tag
            respuesta = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            return respuesta if respuesta else "Movimiento técnico en el tipo de cambio."
        else:
            return "Movimiento técnico en el tipo de cambio."
            
    except:
        return "Ajuste por condiciones macroeconómicas."

# 4. INGESTA DE DATOS (Banxico)
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

# 6. DETECCIÓN DE ANOMALÍAS (Z-Score > 2.5 para ser más selectivos)
ventana_global = Window.partitionBy(F.lit(1))
df_stats = df_features.withColumn("avg_historico", _avg("cambio_diario").over(ventana_global)) \
                       .withColumn("std_historico", _std("cambio_diario").over(ventana_global))

df_final = df_stats.withColumn(
    "z_score_final", 
    (F.col("cambio_diario") - F.col("avg_historico")) / F.col("std_historico")
).withColumn(
    "es_anomalia", 
    when(abs(F.col("z_score_final")) > 2.5, 1).otherwise(0)
)

# 7. INTEGRACIÓN CON IA + PRUEBA HISTÓRICA
noticias_dict = {}

# A. Anomalías actuales detectadas por Spark
anomalias_recientes = df_final.filter(F.col("es_anomalia") == 1) \
                              .sort(F.col("fecha").desc()) \
                              .limit(5).collect()

print(f"Se encontraron {len(anomalias_recientes)} anomalías. Consultando IA...")
for row in anomalias_recientes:
    fecha_str = str(row['fecha'])
    noticias_dict[fecha_str] = obtener_explicacion_ia(fecha_str, row['tipo_cambio'])

# B. PRUEBA DE FUEGO: Forzar fecha histórica para validar a Llama 3
fecha_historica = "2016-11-09"
print(f"🕵️ Realizando prueba de fuego histórica para: {fecha_historica}...")
noticias_dict[fecha_historica] = obtener_explicacion_ia(fecha_historica, 20.50)

# 8. GUARDADO FINAL
df_final.toPandas().to_parquet("datos_anomalias.parquet")

with open("noticias_contexto.json", "w", encoding='utf-8') as f:
    json.dump(noticias_dict, f, ensure_ascii=False, indent=4)

print("✅ Pipeline finalizado con éxito.")
