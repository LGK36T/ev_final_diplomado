# -*- coding: utf-8 -*-
"""
Genera un reporte Markdown consolidado con:
- Info de la serie semanal de casos
- Relación casos vs temperatura (correlaciones)
- Métricas de modelos SARIMAX y SARIMAX+Clima
- Picos de casos y temperatura por ciudad

Salida:
  outputs/report/Model_Report.md
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(".")
FORECAST_DIR = BASE_DIR / "outputs" / "forecast"
RESUMEN_DIR = BASE_DIR / "outputs" / "resumen"
REPORT_DIR = BASE_DIR / "outputs" / "report"     # <--- cambio solicitado
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ Helpers de carga ------------------ #
def load_serie_info(path: Path):
    if not path.exists():
        print(f"[i] No encontré serie semanal en {path}")
        return None

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    start = df.index.min().date()
    end = df.index.max().date()
    n = len(df)
    return {"start": start, "end": end, "n": n}


def load_metricas(path: Path):
    if not path.exists():
        print(f"[i] No encontré métricas en {path}")
        return None

    # Intento lectura con , y fallback con ;
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    # Normalizar nombres de columnas (quita BOM y espacios)
    df.columns = (
        df.columns.astype(str)
          .str.replace("\ufeff", "", regex=False)
          .str.strip()
    )

    cols_lower = [c.lower() for c in df.columns]

    # ---------- Caso A: tabla con columna modelo ----------
    if "modelo" in cols_lower or "model" in cols_lower:
        # Detectar nombre real de la columna modelo
        col_modelo = df.columns[cols_lower.index("modelo")] if "modelo" in cols_lower \
                     else df.columns[cols_lower.index("model")]

        df[col_modelo] = df[col_modelo].astype(str).str.strip()

        base = df[df[col_modelo].str.upper() == "SARIMAX"]
        base_row = base.iloc[0].to_dict() if not base.empty else None

        clima = df[df[col_modelo].str.contains("Clima", case=False, na=False)]
        clima_row = clima.iloc[0].to_dict() if not clima.empty else None

        return {"base": base_row, "clima": clima_row}

    # ---------- Caso B: serie tipo ",0 / MAE,xxx / RMSE,yyy" ----------
    # Si tiene dos columnas y una parece índice de métrica, lo tratamos como serie
    if len(df.columns) == 2:
        # La primera columna suele ser "Unnamed: 0" o similar con MAE/RMSE
        col_metric = df.columns[0]
        col_value = df.columns[1]

        serie = df.set_index(col_metric)[col_value]

        # convertir a dict base
        base_row = {
            "Modelo": "SARIMAX",
            "MAE": float(serie.get("MAE", np.nan)),
            "RMSE": float(serie.get("RMSE", np.nan)),
        }

        return {"base": base_row, "clima": None}

    # Si no calza ningún formato conocido:
    raise ValueError(
        f"Formato de métricas no reconocido. Columnas: {list(df.columns)}"
    )

    if not path.exists():
        print(f"[i] No encontré métricas en {path}")
        return None

    df = pd.read_csv(path)
    df["Modelo"] = df["Modelo"].str.strip()

    base = df[df["Modelo"] == "SARIMAX"]
    base_row = base.iloc[0].to_dict() if not base.empty else None

    clima = df[df["Modelo"].str.contains("Clima", case=False, na=False)]
    clima_row = clima.iloc[0].to_dict() if not clima.empty else None

    return {"base": base_row, "clima": clima_row}


def load_corr_info(path: Path):
    if not path.exists():
        print(f"[i] No encontré correlaciones en {path}")
        return None

    df = pd.read_csv(path)
    row0 = df.loc[df["lag_semanas"] == 0].iloc[0]
    corr0 = float(row0["corr"])

    df["abs_corr"] = df["corr"].abs()
    best = df.loc[df["abs_corr"].idxmax()]
    best_lag = int(best["lag_semanas"])
    best_corr = float(best["corr"])

    return {"corr0": corr0, "best_lag": best_lag, "best_corr": best_corr}


def load_picos_info(path: Path):
    if not path.exists():
        print(f"[i] No encontré picos en {path}")
        return None

    df = pd.read_csv(path)
    resumen = []
    for ciudad in sorted(df["ciudad"].unique()):
        sub = df[df["ciudad"] == ciudad]

        pico_casos = sub[sub["tipo"] == "casos"]
        pico_temp = sub[sub["tipo"] == "temperatura"]

        pc = pico_casos.iloc[0].to_dict() if not pico_casos.empty else None
        pt = pico_temp.iloc[0].to_dict() if not pico_temp.empty else None

        resumen.append({"ciudad": ciudad, "pico_casos": pc, "pico_temp": pt})

    return resumen


# ------------------ Builder del Markdown ------------------ #
def build_markdown(serie_info, metricas, corr_info, picos_info) -> str:

    start = serie_info["start"] if serie_info else "N/D"
    end = serie_info["end"] if serie_info else "N/D"
    n = serie_info["n"] if serie_info else "N/D"

    mae_base = rmse_base = mae_clima = rmse_clima = "N/D"

    if metricas and metricas["base"]:
        mae_base = round(metricas["base"]["MAE"], 2)
        rmse_base = round(metricas["base"]["RMSE"], 2)

    if metricas and metricas["clima"]:
        mae_clima = round(metricas["clima"]["MAE"], 2)
        rmse_clima = round(metricas["clima"]["RMSE"], 2)

    corr0 = best_lag = best_corr = "N/D"

    if corr_info:
        corr0 = round(corr_info["corr0"], 4)
        best_lag = corr_info["best_lag"]
        best_corr = round(corr_info["best_corr"], 4)

    # Tabla de picos
    picos_md_lines = []
    if picos_info:
        picos_md_lines.append("| Ciudad | Tipo | Año-Semana | Fecha | Valor | Temp semana | Casos semana |")
        picos_md_lines.append("|--------|------|------------|-------|--------|-------------|--------------|")

        for item in picos_info:
            ciudad = item["ciudad"]
            pc = item["pico_casos"]
            pt = item["pico_temp"]

            if pc:
                picos_md_lines.append(
                    f"| {ciudad} | Pico casos | {pc['anio']}-{pc['semana']} | "
                    f"{pc['fecha']} | {pc['valor']:.1f} | {pc['temp_en_semana']:.1f}°C | - |"
                )
            if pt:
                picos_md_lines.append(
                    f"| {ciudad} | Pico temp | {pt['anio']}-{pt['semana']} | "
                    f"{pt['fecha']} | {pt['valor']:.1f}°C | - | {pt['casos_en_semana']:.1f} |"
                )

        picos_md = "\n".join(picos_md_lines)
    else:
        picos_md = "_No se encontraron picos_"

    # --- Markdown completo --- #
    md = f"""
# Model Report — Urgencias Respiratorias (Región de Antofagasta)

**Propósito:** pronosticar casos semanales para apoyar planificación sanitaria incorporando estacionalidad y clima.

---

## 1. Datos

- Serie semanal procesada: **{start} → {end}** ({n} semanas).
- Frecuencia: semanal (`W-MON`).
- Clima semanal: temperatura promedio por ciudad (Antofagasta/Calama).

---

## 2. Relación Casos ↔ Temperatura (Antofagasta)

- Correlación lag 0: **{corr0}**
- Máxima correlación (|corr|) en lag **{best_lag}** → **{best_corr}**

Interpretación: existe relación negativa moderada → semanas más frías → más casos.

---

## 3. Picos por Ciudad

{picos_md}

---

## 4. Modelos SARIMAX

### Métricas en holdout (últimas 52 semanas)

| Modelo         | MAE    | RMSE   |
|----------------|--------|--------|
| SARIMAX        | {mae_base} | {rmse_base} |
| SARIMAX+Clima  | {mae_clima} | {rmse_clima} |

Interpretación:
- El modelo baseline es mejor.
- La temperatura no mejora el pronóstico directo.

---

## 5. Interpretabilidad

- La estacionalidad de 52 semanas captura el ciclo invierno–verano.
- El clima influye, pero su efecto ya está incorporado en la estacionalidad.
- IC95% reflejan incertidumbre creciente hacia el futuro.

---

## 6. Riesgos y Limitaciones

- Posible subregistro de casos.
- Cambios estructurales (COVID, flujos de demanda).
- Clima simplificado → solo temperatura promedio.
- SARIMAX es lineal → puede perder relaciones no lineales.

---

## 7. Recomendaciones

- Monitorear semanas críticas (26–32).
- Reentrenar mensualmente.
- Incorporar:
  - temperatura mínima/máxima  
  - humedad  
  - circulación viral (influenza/RSV)  
  - movilidad

---

## 8. Próximos Pasos

- Probar modelos no lineales (GAM, RF, GBM).
- Dashboard automatizado (Power BI / Superset).
- API interna para consulta de pronósticos.

---

Fin del reporte.
"""

    return md


# ------------------ MAIN ------------------ #
def main():
    serie_info = load_serie_info(FORECAST_DIR / "serie_semanal_limpia.csv")
    metricas = load_metricas(FORECAST_DIR / "metricas_forecast.csv")
    corr_info = load_corr_info(FORECAST_DIR / "correlacion_casos_temp_lags.csv")
    picos_info = load_picos_info(RESUMEN_DIR / "picos_casos_temperatura.csv")

    md_text = build_markdown(serie_info, metricas, corr_info, picos_info)

    out_file = REPORT_DIR / "Model_Report.md"
    out_file.write_text(md_text, encoding="utf-8")

    print(f"✅ Reporte Markdown generado en: {out_file}")


if __name__ == "__main__":
    main()

