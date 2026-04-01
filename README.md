# mx-anomaly-detector
## El Proyecto ##
Este sistema monitorea en tiempo real el tipo de cambio USD/MXN utilizando la API de Banxico. No es una simple gráfica; es un ecosistema que detecta automáticamente crisis financieras o movimientos atípicos (anomalías) mediante estadística avanzada y procesamiento distribuido.

## Arquitectura del Sistema ##
El proyecto implementa un flujo de ingenieria de datos y automatización.  

**Ingesta:** Conexión segura a la API de Banxico (SIE) para obtener datos históricos y actuales del tipo de cambio.  

**Procesamiento (Apache Spark):** Limpieza de datos y tipado fuerte con PySpark.  

**Feature Engineering:** Cálculo de medias móviles, desviaciones estándar y deltas diarios.  

**Detección de Anomalías:** Implementación estadística de Z-Score: $$Z = \frac{x - \mu}{\sigma}$$ Definimos una anomalía cuando el cambio diario se aleja más de 3 desviaciones estándar del promedio histórico.  

**Almacenamiento:** Los resultados se exportan en formato Parquet, optimizando el espacio en disco y la velocidad de lectura para la App.  

**Orquestación (GitHub Actions):** Un "robot" (workflow CI/CD) ejecuta el clúster de Spark diariamente a las 08:00 AM CDMX.  

**Visualización (Streamlit):** Dashboard interactivo alojado en la nube que permite explorar cada anomalía detectada con gráficos dinámicos de Plotly.
