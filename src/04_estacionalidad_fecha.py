# -*- coding: utf-8 -*-
"""
Pronóstico semanal con SARIMAX (estacionalidad ~52)
- Lee outputs/eda/casos_por_semana.csv (fecha o año+semana)
- Split temporal: train / test (últimas 52 semanas)
- Métricas: MAE y RMSE
- Gráfico: real vs predicho + forecast 26 semanas hacia delante
Salidas en outputs/forecast/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURACIÓN DE RUTAS ---
EDA_DIR = Path("outputs/eda")
OUT_DIR = Path("outputs/forecast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FUNCIONES AUXILIARES ---
def _yws_to_date(y, w):
    """Convierte año y semana a una fecha de inicio de semana."""
    try:
        return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except Exception:
        return pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)

def cargar_serie():
    """Carga la serie de tiempo desde el archivo generado por el EDA."""
    p = EDA_DIR / "casos_por_semana.csv"
    if not p.exists():
        raise FileNotFoundError("Falta outputs/eda/casos_por_semana.csv. Ejecuta antes el EDA.")
    
    df = pd.read_csv(p)
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    fcol = next((c for c in cols if "fecha" in c), None)
    if fcol:
        try:
            df[fcol] = pd.to_datetime(df[fcol], errors="coerce")
            if df[fcol].notna().sum() > 10 and df[fcol].nunique() > 10:
                print("[i] Serie de tiempo cargada usando columna de fecha.")
                return df.rename(columns={fcol: "fecha_sem", df.columns[1]: "casos"})[["fecha_sem", "casos"]]
        except Exception:
            pass

    acol = next((c for c in ["anio","año","ano","year"] if c in cols), None)
    scol = next((c for c in ["semana", "semanaestadistica", "semana_epidemiologica", "nsemana"] if c in cols), None)
    
    if acol and scol:
        print(f"[i] Serie de tiempo cargada usando columnas: año='{acol}', semana='{scol}'.")
        tcol = next((c for c in df.columns if c not in [acol, scol]), None)
        s = df[[acol, scol, tcol]].dropna().copy()
        s["fecha_sem"] = s.apply(lambda r: _yws_to_date(r[acol], r[scol]), axis=1)
        s = s.rename(columns={tcol: "casos"})
        return s[["fecha_sem", "casos"]]

    raise ValueError("No pude interpretar casos_por_semana.csv (esperaba 'fecha' o 'año'+'semana').")

def preparar_serie_sem(s):
    """Limpia y prepara la serie para el modelo de pronóstico."""
    s = s.copy().sort_values("fecha_sem")
    y = (s.set_index("fecha_sem")["casos"]
          .resample("W-MON").sum()
          .asfreq("W-MON")
          .interpolate())
    
    y = y.fillna(0)
    y[y < 0] = 0
    return y

def entrenar_y_evaluar(y, h_test=52, seasonal_period=52):
    """Entrena un modelo SARIMAX y evalúa su rendimiento."""
    if len(y) <= h_test + seasonal_period:
        h_test = max(4, min(12, len(y) // 4))

    y_train, y_test = y.iloc[:-h_test], y.iloc[-h_test:]
    
    order = (1, 1, 1)
    seasonal_order = (1, 1, 0, seasonal_period)
    
    print(f"Entrenando modelo SARIMAX({order}, {seasonal_order})...")
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=h_test)
    y_pred = pred.predicted_mean
    mae = mean_absolute_error(y_test, y_pred)
    
    # ----- LÍNEA CORREGIDA -----
    # Se calcula la raíz cuadrada del MSE para obtener el RMSE.
    # Esto es compatible con todas las versiones de scikit-learn.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    f_steps = 26
    future_pred = res.get_forecast(steps=h_test + f_steps)
    y_future = future_pred.predicted_mean.iloc[-f_steps:]

    return res, (y_train, y_test, y_pred), (mae, rmse), y_future

def graficar(y_train, y_test, y_pred, y_future, out_png):
    """Genera y guarda el gráfico del pronóstico."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 5))
    
    plt.plot(y_train.index, y_train.values, label="Datos de Entrenamiento", color='royalblue', lw=1.5)
    plt.plot(y_test.index, y_test.values, label="Datos Reales (Test)", color='darkgreen', lw=2)
    plt.plot(y_pred.index, y_pred.values, label="Predicción en Test", color='darkorange', lw=2, linestyle='--')
    if y_future is not None:
        plt.plot(y_future.index, y_future.values, label="Pronóstico a 26 Semanas", color='red', lw=2, linestyle='--')
    
    plt.title("Pronóstico SARIMAX de Casos Semanales", fontsize=14)
    plt.xlabel("Fecha", fontsize=10); plt.ylabel("Número de Casos", fontsize=10)
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    """Orquesta la ejecución del script."""
    print("[1/3] Cargando y preparando la serie semanal…")
    s = cargar_serie()
    y = preparar_serie_sem(s)
    y.to_csv(OUT_DIR / "serie_semanal_limpia.csv")
    print(f"Serie lista: {y.index.min().date()} → {y.index.max().date()} | n={len(y)} puntos")

    print("\n[2/3] Entrenando modelo SARIMAX y evaluando…")
    res, (y_tr, y_te, y_pr), (mae, rmse), y_fut = entrenar_y_evaluar(y)
    
    metrics = pd.Series({"MAE": mae, "RMSE": rmse})
    metrics.to_csv(OUT_DIR / "metricas_forecast.csv")
    print(f"Resultados en conjunto de prueba: MAE={mae:.2f} | RMSE={rmse:.2f}")

    print("\n[3/3] Generando gráfico del pronóstico…")
    graficar(y_tr, y_te, y_pr, y_fut, OUT_DIR / "forecast_sarimax.png")
    
    print(f"\n✅ Proceso completado. Revisa la carpeta: '{OUT_DIR}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] ERROR: El script falló. Causa: {e}")