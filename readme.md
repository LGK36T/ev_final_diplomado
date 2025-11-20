# eva3_mod5 — Pipeline EDA/ETL/Forecast (Urgencias Respiratorias)

Proyecto de análisis, modelamiento y pronóstico de casos semanales de urgencias respiratorias para la Región de Antofagasta, incorporando variables climáticas y análisis de estacionalidad.  
La infra está pensada como **pipeline reproducible**: los scripts en `src/` se ejecutan en orden numérico y generan artefactos en `outputs/`.

---

## Estructura del repositorio

eva3_mod5/
├─ artefactos/ # recursos auxiliares / entregables
├─ data/ # datos de entrada (raw o semiprocesados)
├─ notebooks/ # exploración en notebooks (opcional)
├─ outputs/ # salidas generadas por el pipeline
├─ src/ # scripts principales en orden de ejecución
│ ├─ 01_eda_urgencias.py
│ ├─ 02_etl_dims_fact.py
│ ├─ 03_etl_clima.py
│ ├─ 04_estacionalidad_fecha.py
│ ├─ 05_estacionalidad_clima.py
│ ├─ 06_correlacion_ciudad.py
│ ├─ 07_model_selection.py
│ ├─ 08_interpretabilidad_etica.py
│ ├─ 09_resumen_estacionalidad.py
│ ├─ 10_resumen_estacionalidad_detecta_picos.py
│ ├─ 11_model_card.py
│ ├─ 12_update_forecast.py
│ ├─ 13_regresion.py
│ └─ 14_Dim_Tablas.py
├─ utils/
│ └─ show_datasets.py # visualización/resumen final de datasets
├─ venv/ # entorno virtual (autogenerado)
├─ execute.py # ejecutor del pipeline end-to-end
├─ README.md
└─ requirements.txt


---

## Requisitos

- Python 3.10+ (recomendado).
- Dependencias listadas en `requirements.txt`.

**Nota:** `execute.py` está hecho para ser “one-shot”:  
si no existe `venv/`, lo crea; si faltan librerías, las instala.

---

## Cómo ejecutar el proyecto

### Opción recomendada (pipeline completo)
Desde la raíz del repo:

```bash
python3 execute.py

Esto hará:

Crear venv/ si no existe.

Instalar dependencias (desde requirements.txt).

Ejecutar todos los scripts de src/ en orden.

Ejecutar utils/show_datasets.py al final.

Orden del pipeline (src/)

Los scripts están numerados para garantizar la ejecución secuencial.
A continuación se describe qué produce cada etapa a alto nivel.

01_eda_urgencias.py — EDA inicial de urgencias

Limpieza básica, validación de columnas.

Genera primeras salidas exploratorias (resúmenes y gráficos).

02_etl_dims_fact.py — ETL a modelo dimensional

Construcción de tablas tipo dimensiones/fact.

Normalización y consistencia de claves.

03_etl_clima.py — ETL de clima

Limpieza y agregación de clima (temperatura, etc.).

Deja clima alineado en frecuencia semanal.

04_estacionalidad_fecha.py — Estacionalidad en serie principal

Descomposición/diagnóstico estacional de casos.

Genera artefactos de estacionalidad temporal.

05_estacionalidad_clima.py — Estacionalidad en clima

Análisis estacional de variables meteorológicas.

06_correlacion_ciudad.py — Correlación casos vs clima

Correlaciones por lag y por ciudad.

Salida típica: outputs/forecast/correlacion_casos_temp_lags.csv.

07_model_selection.py — Selección de modelos

Entrena y evalúa candidatos SARIMAX / SARIMAX + clima.

Genera métricas y elección de baseline.

08_interpretabilidad_etica.py — Interpretabilidad/ética

Revisión de supuestos, sesgos, límites del modelo.

09_resumen_estacionalidad.py — Reporte consolidado

Construye un reporte Markdown con:

resumen de serie

correlaciones

métricas

picos por ciudad

Salida: outputs/report/Model_Report.md.

10_resumen_estacionalidad_detecta_picos.py — Detección de picos

Detecta semanas extremas de casos y temperatura.

Salida típica: outputs/resumen/picos_casos_temperatura.csv.

11_model_card.py — Model card

Documento de modelo (propósito, datos, métricas, limitaciones).

12_update_forecast.py — Pronóstico final

Entrenamiento final y generación de forecast a futuro.

Produce serie pronosticada y gráficos.

13_regresion.py — Modelos/chequeos alternativos

Regresiones complementarias o pruebas auxiliares.

14_Dim_Tablas.py — Consolidación final de dimensiones

Ajustes finales de tablas, export y consistencia.

Final: utils/show_datasets.py

Muestra/valida datasets generados en outputs/.

Checklist visual para la entrega.

Outputs esperados (resumen)

Dependiendo de los datos de entrada, el pipeline produce:

outputs/forecast/

serie_semanal_limpia.csv

metricas_forecast.csv

correlacion_casos_temp_lags.csv

pronósticos y gráficos

outputs/resumen/

picos_casos_temperatura.csv

resúmenes intermedios

outputs/report/

Model_Report.md

Autores y contribuyentes

Autores principales

- Fabián Araya
- Felipe Venegas
- Seba