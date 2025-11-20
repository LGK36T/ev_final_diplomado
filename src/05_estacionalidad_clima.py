# -*- coding: utf-8 -*-
"""
Pronóstico semanal con SARIMAX (estacionalidad ~52) + análisis de relación con clima.

Hace:
- Lee outputs/eda/casos_por_semana.csv (fecha o año+semana).
- Prepara serie semanal 'casos' (W-MON).
- Carga clima semanal desde data/clima_anio_semana.csv (Antofagasta).
- Cruza casos vs temperatura:
    * Correlación simple.
    * Correlaciones con lags.
    * Gráfico scatter casos vs temperatura.
    * Gráfico de estacionalidad (promedio por semana del año).
- Entrena:
    * SARIMAX baseline (sin exógenas).
    * SARIMAX+Clima (temperatura como regresor).
- Exporta métricas en outputs/forecast/metricas_forecast.csv
- Exporta artefactos visuales en outputs/forecast/

Requiere:
- outputs/eda/casos_por_semana.csv
- data/clima_anio_semana.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- RUTAS ---
EDA_DIR = Path("outputs") / "eda"
OUT_DIR = Path("outputs") / "forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ruta del archivo de clima ya agregado por año + semana
CLIMA_SEM_PATH = Path("data") / "clima_anio_semana.csv"


# ---------------------------------------------------------------------
# FUNCIONES AUXILIARES SERIE PRINCIPAL
# ---------------------------------------------------------------------
def _yws_to_date(y, w):
    """Convierte año y semana a una fecha de inicio de semana (lunes)."""
    try:
        return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except Exception:
        # Fallback: año + semanas como aproximación
        return pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w) - 1)


def cargar_serie():
    """
    Carga la serie de tiempo desde outputs/eda/casos_por_semana.csv.

    Soporta dos formatos:
    - Columna única de fecha + columna de casos
    - Columnas de año + semana + casos
    """
    p = EDA_DIR / "casos_por_semana.csv"
    if not p.exists():
        raise FileNotFoundError(
            "Falta outputs/eda/casos_por_semana.csv. Ejecuta antes el EDA."
        )

    df = pd.read_csv(p)
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # 1) Buscar columna de fecha
    fcol = next((c for c in cols if "fecha" in c), None)
    if fcol:
        try:
            df[fcol] = pd.to_datetime(df[fcol], errors="coerce")
            if df[fcol].notna().sum() > 10 and df[fcol].nunique() > 10:
                print("[i] Serie de tiempo cargada usando columna de fecha.")
                otras = [c for c in df.columns if c != fcol]
                if not otras:
                    raise ValueError(
                        "No encontré columna de casos junto a la fecha."
                    )
                tcol = otras[0]
                df = df.rename(columns={fcol: "fecha_sem", tcol: "casos"})
                return df[["fecha_sem", "casos"]]
        except Exception:
            pass

    # 2) Buscar año + semana
    acol = next((c for c in ["anio", "año", "ano", "year"] if c in cols), None)
    scol = next(
        (
            c
            for c in [
                "semana",
                "semanaestadistica",
                "semana_epidemiologica",
                "nsemana",
            ]
            if c in cols
        ),
        None,
    )

    if acol and scol:
        print("[i] Serie de tiempo cargada usando año + semana.")
        s = df.copy()
        candidatos = [c for c in s.columns if c not in (acol, scol)]
        if not candidatos:
            raise ValueError(
                "No encontré columna de casos junto a año y semana."
            )
        tcol = candidatos[0]
        s["fecha_sem"] = s.apply(
            lambda r: _yws_to_date(r[acol], r[scol]), axis=1
        )
        s = s.rename(columns={tcol: "casos"})
        return s[["fecha_sem", "casos"]]

    raise ValueError(
        "No pude interpretar casos_por_semana.csv "
        "(esperaba 'fecha' o 'año'+'semana')."
    )


def preparar_serie_sem(s: pd.DataFrame) -> pd.Series:
    """Limpia y prepara la serie semanal para el modelo de pronóstico."""
    s = s.copy().sort_values("fecha_sem")
    y = (
        s.set_index("fecha_sem")["casos"]
        .resample("W-MON")
        .sum()
        .asfreq("W-MON")
        .interpolate()
    )
    # Seguridad: sin NaN y sin negativos
    y = y.fillna(0)
    y[y < 0] = 0
    return y


# ---------------------------------------------------------------------
# CLIMA DESDE clima_anio_semana.csv
# ---------------------------------------------------------------------
def cargar_clima_desde_anio_semana(y_index: pd.DatetimeIndex,
                                   ciudad: str = "Antofagasta"):
    """
    Carga temperaturas desde data/clima_anio_semana.csv,
    que viene en formato (ciudad, anio, semanaestadistica, temperatura),
    genera fecha_sem con la misma convención que la serie y,
    y alinea temperatura al índice de y (W-MON).

    Retorna DataFrame con:
      - índice = y_index
      - columna 'temp_promedio'
    """
    if not CLIMA_SEM_PATH.exists():
        raise FileNotFoundError(
            f"No encontré archivo de clima semanal en {CLIMA_SEM_PATH}"
        )

    clima = pd.read_csv(CLIMA_SEM_PATH)

    # Normalizar nombres esperados
    clima.columns = [c.strip().lower() for c in clima.columns]

    # Chequear columnas mínimas
    required = {"ciudad", "anio", "semanaestadistica", "temperatura"}
    if not required.issubset(set(clima.columns)):
        raise ValueError(
            f"Se esperaban columnas {required}, pero encontré: {set(clima.columns)}"
        )

    clima = clima[clima["ciudad"] == ciudad].copy()
    if clima.empty:
        raise ValueError(
            f"No hay registros de clima para ciudad='{ciudad}' en {CLIMA_SEM_PATH}"
        )

    # Construir fecha_sem compatible usando el helper
    clima["fecha_sem"] = clima.apply(
        lambda r: _yws_to_date(r["anio"], r["semanaestadistica"]), axis=1
    )

    clima = (
        clima[["fecha_sem", "temperatura"]]
        .groupby("fecha_sem")["temperatura"]
        .mean()
        .sort_index()
    )

    # Alinear a índice de y
    clima_aligned = clima.reindex(y_index).interpolate(
        limit_direction="both"
    )
    clima_aligned = clima_aligned.ffill().bfill()

    return clima_aligned.to_frame(name="temp_promedio")


# ---------------------------------------------------------------------
# ANÁLISIS DE RELACIÓN Y ESTACIONALIDAD
# ---------------------------------------------------------------------
def analizar_relacion_y_estacionalidad(y: pd.Series,
                                       temp_df: pd.DataFrame,
                                       out_dir: Path):
    """
    Calcula:
      - Correlación simple casos vs temperatura.
      - Correlación con lags (-8 a +8 semanas).
      - Gráfico scatter casos vs temperatura.
      - Gráfico de estacionalidad promedio por semana del año.

    Guarda:
      - correlacion_casos_temp_lags.csv
      - scatter_casos_vs_temp.png
      - estacionalidad_casos_temp.png
    """
    temp = temp_df["temp_promedio"]

    df_merge = pd.DataFrame({
        "casos": y,
        "temp": temp
    }).dropna()

    # Correlación simple
    corr_0 = df_merge["casos"].corr(df_merge["temp"])
    print(f"\n[Relación casos vs temperatura] Correlación (lag 0): {corr_0:.4f}")

    # Correlaciones por lag
    registros_lag = []
    for lag in range(-8, 9):  # -8..+8 semanas
        if lag < 0:
            # temp adelantada: casos vs temp(t+lag)
            corr = df_merge["casos"].corr(df_merge["temp"].shift(-lag))
        else:
            corr = df_merge["casos"].corr(df_merge["temp"].shift(lag))
        registros_lag.append({"lag_semanas": lag, "corr": corr})

    corr_df = pd.DataFrame(registros_lag)
    corr_df.to_csv(out_dir / "correlacion_casos_temp_lags.csv", index=False)
    print("\nCorrelaciones por lag (guardadas en correlacion_casos_temp_lags.csv):")
    print(corr_df)

    # Scatter casos vs temperatura
    plt.figure(figsize=(6, 5))
    plt.scatter(df_merge["temp"], df_merge["casos"], alpha=0.5)
    plt.xlabel("Temperatura promedio semanal (°C)")
    plt.ylabel("Casos semanales")
    plt.title("Casos vs Temperatura semanal")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_casos_vs_temp.png", dpi=150)
    plt.close()

    # Estacionalidad por semana del año (usando índice de y)
    est = pd.DataFrame({
        "casos": y,
        "temp": temp
    }).dropna()
    est["semana"] = est.index.isocalendar().week

    est_agg = est.groupby("semana").agg(
        casos_promedio=("casos", "mean"),
        temp_promedio=("temp", "mean")
    ).reset_index()

    # Normalizar para graficar en misma escala
    casos_norm = (est_agg["casos_promedio"] - est_agg["casos_promedio"].mean()) / est_agg["casos_promedio"].std()
    temp_norm = (est_agg["temp_promedio"] - est_agg["temp_promedio"].mean()) / est_agg["temp_promedio"].std()

    plt.figure(figsize=(10, 5))
    plt.plot(est_agg["semana"], casos_norm, label="Casos (normalizado)")
    plt.plot(est_agg["semana"], temp_norm, label="Temperatura (normalizada)")
    plt.xlabel("Semana del año")
    plt.ylabel("Valor normalizado (Z-score)")
    plt.title("Estacionalidad semanal: Casos vs Temperatura")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "estacionalidad_casos_temp.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# ENTRENAMIENTO Y GRÁFICO SARIMAX
# ---------------------------------------------------------------------
def entrenar_y_evaluar(
    y: pd.Series,
    h_test: int = 52,
    seasonal_period: int = 52,
    exog: pd.DataFrame | None = None,
):
    """
    Entrena un modelo SARIMAX y evalúa su rendimiento.
    - y: serie objetivo (index datetime)
    - exog: DataFrame alineado con y (mismo índice) o None
    Retorna:
      res (resultado statsmodels),
      (y_train, y_test, y_pred),
      (mae, rmse),
      y_future (solo para baseline sin exógenas)
    """
    if len(y) <= h_test + seasonal_period:
        h_test = max(4, min(12, len(y) // 4))

    y_train, y_test = y.iloc[:-h_test], y.iloc[-h_test:]

    if exog is not None:
        exog_train = exog.iloc[:-h_test]
        exog_test = exog.iloc[-h_test:]
    else:
        exog_train = None
        exog_test = None

    order = (1, 1, 1)
    seasonal_order = (1, 1, 0, seasonal_period)

    print(
        f"Entrenando modelo SARIMAX({order}, {seasonal_order})"
        f"{' + exógenas' if exog is not None else ''}..."
    )
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Predicción en test
    pred = res.get_forecast(steps=h_test, exog=exog_test)
    y_pred = pred.predicted_mean

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Pronóstico futuro solo para baseline (sin exógenas)
    y_future = None
    if exog is None:
        f_steps = 26
        future_pred = res.get_forecast(steps=h_test + f_steps)
        y_future = future_pred.predicted_mean.iloc[-f_steps:]

    return res, (y_train, y_test, y_pred), (mae, rmse), y_future


def graficar(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_future: pd.Series | None,
    out_png: Path,
):
    """Genera y guarda el gráfico del pronóstico."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 5))

    plt.plot(
        y_train.index,
        y_train.values,
        label="Datos de Entrenamiento",
        color="royalblue",
        lw=1.5,
    )
    plt.plot(
        y_test.index,
        y_test.values,
        label="Datos Reales (Test)",
        color="darkgreen",
        lw=2,
    )
    plt.plot(
        y_pred.index,
        y_pred.values,
        label="Predicción en Test",
        color="darkorange",
        lw=2,
        linestyle="--",
    )
    if y_future is not None:
        plt.plot(
            y_future.index,
            y_future.values,
            label="Pronóstico a 26 Semanas",
            color="red",
            lw=2,
            linestyle="--",
        )

    plt.title("Pronóstico SARIMAX de Casos Semanales", fontsize=14)
    plt.xlabel("Fecha", fontsize=10)
    plt.ylabel("Número de Casos", fontsize=10)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    """Orquesta la ejecución del script."""
    print("[1/4] Cargando y preparando la serie semanal de casos…")
    s = cargar_serie()
    y = preparar_serie_sem(s)
    y.to_csv(OUT_DIR / "serie_semanal_limpia.csv")
    print(
        f"Serie lista: {y.index.min().date()} → {y.index.max().date()} | "
        f"n={len(y)} puntos"
    )

    print("\n[2/4] Cargando clima semanal y analizando relación...")
    temp_df = cargar_clima_desde_anio_semana(y.index, ciudad="Antofagasta")
    analizar_relacion_y_estacionalidad(y, temp_df, OUT_DIR)

    print("\n[3/4] Entrenando modelos SARIMAX (baseline y con clima)…")
    # 3a) Modelo baseline sin exógenas
    res_base, (y_tr, y_te, y_pr), (mae_base, rmse_base), y_fut = entrenar_y_evaluar(y)

    # 3b) Modelo con clima como exógeno
    mae_exog = rmse_exog = None
    try:
        res_exog, (_, _, y_pr_exog), (mae_exog, rmse_exog), _ = entrenar_y_evaluar(
            y, exog=temp_df
        )
        print(
            f"Resultados SARIMAX+Clima: "
            f"MAE={mae_exog:.2f} | RMSE={rmse_exog:.2f}"
        )
    except Exception as e:
        print(f"[i] Error en SARIMAX+Clima: {e}")

    # 4) Guardar métricas comparativas
    print("\n[4/4] Guardando métricas y gráfico de forecast…")
    rows = [{"Modelo": "SARIMAX", "MAE": mae_base, "RMSE": rmse_base}]
    if mae_exog is not None and rmse_exog is not None:
        rows.append(
            {"Modelo": "SARIMAX+Clima", "MAE": mae_exog, "RMSE": rmse_exog}
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metricas_forecast.csv", index=False)
    print("\nMétricas comparativas:")
    print(metrics_df)

    graficar(y_tr, y_te, y_pr, y_fut, OUT_DIR / "forecast_sarimax.png")

    print(f"\n✅ Proceso completado. Revisa la carpeta: '{OUT_DIR}'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] ERROR: El script falló. Causa: {e}")
