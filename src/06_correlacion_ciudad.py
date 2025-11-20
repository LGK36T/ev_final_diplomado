# -*- coding: utf-8 -*-
"""
Correlación clima–casos por ciudad (Antofagasta, Calama).

Lee:
- data/clima_anio_semana.csv  (ciudad, anio, semanaestadistica, temperatura)
- outputs/eda/casos_por_semana.csv  (anio, semanaestadistica, casos)

Hace:
- Prepara la serie semanal de casos en formato W-MON.
- Prepara clima semanal por ciudad.
- Calcula:
    * correlación simple (lag 0)
    * correlaciones para lags -8..+8
- Genera:
    * correlacion_ciudad_<c>.csv
    * scatter_ciudad_<c>.png
    * estacionalidad_ciudad_<c>.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CLIMA_PATH = Path("data") / "clima_anio_semana.csv"
CASOS_PATH = Path("outputs") / "eda" / "casos_por_semana.csv"
OUT_DIR = Path("outputs") / "correlacion_ciudades"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------
# Helpers
# --------------------------------------------
def yws_to_date(y, w):
    """Convierte año + semana a fecha lunes."""
    try:
        return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except:
        return pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)


def preparar_casos(df):
    df = df.copy()
    df["fecha"] = df.apply(lambda r: yws_to_date(r["anio"], r["semanaestadistica"]), axis=1)
    s = (
        df.set_index("fecha")["casos"]
        .resample("W-MON")
        .sum()
        .asfreq("W-MON")
        .interpolate()
    )
    s = s.fillna(0)
    return s


def preparar_clima(df, ciudad):
    dfc = df[df["ciudad"] == ciudad].copy()
    if dfc.empty:
        raise ValueError(f"No hay clima para ciudad: {ciudad}")

    dfc["fecha"] = dfc.apply(lambda r: yws_to_date(r["anio"], r["semanaestadistica"]), axis=1)
    s = (
        dfc.set_index("fecha")["temperatura"]
        .resample("W-MON")
        .mean()
        .asfreq("W-MON")
        .interpolate()
    )
    s = s.ffill().bfill()
    return s


def correlacion_ciudad(casos, clima, ciudad):
    df = pd.DataFrame({"casos": casos, "temp": clima}).dropna()

    # Correlación simple
    corr0 = df["casos"].corr(df["temp"])
    print(f"\n[{ciudad}] Correlación lag 0 = {corr0:.4f}")

    # Correlaciones por lag ±8
    rows = []
    for lag in range(-8, 9):
        if lag < 0:
            c = df["casos"].corr(df["temp"].shift(-lag))
        else:
            c = df["casos"].corr(df["temp"].shift(lag))
        rows.append({"lag": lag, "corr": c})

    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(OUT_DIR / f"correlacion_lags_{ciudad}.csv", index=False)

    # Scatter
    plt.figure(figsize=(6,5))
    plt.scatter(df["temp"], df["casos"], alpha=0.5)
    plt.xlabel("Temperatura semanal")
    plt.ylabel("Casos semanales")
    plt.title(f"Casos vs Temperatura – {ciudad}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"scatter_{ciudad}.png", dpi=150)
    plt.close()

    # Estacionalidad
    df["semana"] = df.index.isocalendar().week
    est = df.groupby("semana").agg(
        casos_prom=("casos","mean"),
        temp_prom=("temp","mean")
    )

    casos_norm = (est["casos_prom"] - est["casos_prom"].mean()) / est["casos_prom"].std()
    temp_norm = (est["temp_prom"] - est["temp_prom"].mean()) / est["temp_prom"].std()

    plt.figure(figsize=(10,5))
    plt.plot(est.index, casos_norm, label="Casos (Z-score)")
    plt.plot(est.index, temp_norm, label="Temp (Z-score)")
    plt.title(f"Estacionalidad semanal – {ciudad}")
    plt.xlabel("Semana")
    plt.ylabel("Valor normalizado")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"estacionalidad_{ciudad}.png", dpi=150)
    plt.close()

    return corr0, lag_df


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():

    # 1. Cargar datos
    clima = pd.read_csv(CLIMA_PATH)
    clima.columns = clima.columns.str.lower()

    casos = pd.read_csv(CASOS_PATH)
    casos.columns = casos.columns.str.lower()

    # Encontrar la columna de casos
    caso_col = next(c for c in casos.columns if c not in ("anio","semanaestadistica"))

    casos = casos.rename(columns={caso_col: "casos"})

    # Crear serie semanal de casos
    serie_casos = preparar_casos(casos)

    # Ciudades disponibles en clima
    ciudades = clima["ciudad"].unique()

    print("\nCiudades detectadas en el clima:", ciudades)

    # 2. Correr análisis por ciudad
    for c in ciudades:
        print(f"\n=== Analizando ciudad: {c} ===")

        serie_temp = preparar_clima(clima, c)

        corr0, lag_df = correlacion_ciudad(serie_casos, serie_temp, c)

        print(f"[{c}] Correlación simple: {corr0:.4f}")
        print(f"[{c}] Archivo de correlación por lag generado.")

    print("\n✔ Proceso completado. Revisa outputs/correlacion_ciudades/")


if __name__ == "__main__":
    main()
