import os
import requests
import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_date, abs, when, avg as _avg, stddev as _std

# 1. INICIALIZAR SPARK
spark = SparkSession.builder.master("local[*]").appName("AutomatedSparkPipeline").getOrCreate()

# 2. CONFIGURACIÓN DE API
TOKEN = os.getenv("BANXICO_TOKEN") 
SERIE = 'SF43718' 
URL = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/{SERIE}/datos'

# 3. FUNCIÓN DE IA (Con la URL que el Router SI acepta)
from huggingface_hub import InferenceClient

def obtener_explicacion_ia(fecha, valor):
    token = os.getenv("HF_TOKEN")
    client = InferenceClient(api_key=token)
    
    # --- PROMPT RECARGADO EN ESPAÑOL ---
    prompt = (
        f"Eres un analista financiero experto. Explica en UNA SOLA oración corta y EN ESPAÑOL "
        f"qué evento causó la volatilidad del peso mexicano el {fecha}. "
        f"No uses inglés, responde solo en español."
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                {"role": "system", "content": "Responde siempre en español mexicano de forma profesional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80, # Le damos un poco más de espacio porque el español es más largo
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"DEBUG [{fecha}]: Error - {str(e)}")
        return "Variación por volatilidad del mercado."

# 4. INGESTA DE DATOS
headers = {'Bmx-Token': TOKEN}
response = requests.get(URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    observations = data['bmx']['series'][0]['datos']
    pdf = pd.DataFrame(observations)
    pdf['dato'] = pd.to_numeric(pdf['dato'])
    df_spark = spark.createDataFrame(pdf)
    df_spark = df_spark.withColumn("fecha", to_date(col("fecha"), "dd/MM/yyyy")).withColumnRenamed("dato", "tipo_cambio")
else:
    raise Exception(f"Error Banxico: {response.status_code}")

# 5. FEATURE ENGINEERING
windowSpec = Window.orderBy("fecha")
df_features = df_spark.withColumn("precio_anterior", F.lag("tipo_cambio", 1).over(windowSpec)) \
    .withColumn("cambio_diario", F.col("tipo_cambio") - F.col("precio_anterior")).na.drop()

# 6. DETECCIÓN DE ANOMALÍAS
ventana_global = Window.partitionBy(F.lit(1))
df_stats = df_features.withColumn("avg_historico", _avg("cambio_diario").over(ventana_global)) \
                       .withColumn("std_historico", _std("cambio_diario").over(ventana_global))

df_final = df_stats.withColumn(
    "z_score_final", (F.col("cambio_diario") - F.col("avg_historico")) / F.col("std_historico")
).withColumn(
    "es_anomalia", when(abs(F.col("z_score_final")) > 2.5, 1).otherwise(0)
)

# 7. INTEGRACIÓN CON IA + PRUEBA HISTÓRICA
noticias_dict = {}
anomalias_recientes = df_final.filter(F.col("es_anomalia") == 1).sort(F.col("fecha").desc()).limit(5).collect()

for row in anomalias_recientes:
    fecha_str = str(row['fecha'])
    noticias_dict[fecha_str] = obtener_explicacion_ia(fecha_str, row['tipo_cambio'])

# PRUEBA DE FUEGO (TRUMP 2016)
fecha_historica = "2016-11-09"
noticias_dict[fecha_historica] = obtener_explicacion_ia(fecha_historica, 20.50)

# 8. GUARDADO FINAL
df_final.toPandas().to_parquet("datos_anomalias.parquet")
with open("noticias_contexto.json", "w", encoding='utf-8') as f:
    json.dump(noticias_dict, f, ensure_ascii=False, indent=4)

print("✅ Pipeline finalizado: noticias_contexto.json actualizado.")
