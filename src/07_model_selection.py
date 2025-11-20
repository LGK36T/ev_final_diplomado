# -*- coding: utf-8 -*-
"""
Selección de modelo temporal:
- Grid SARIMAX (varias combinaciones) + baseline Naïve-52
- Prophet opcional (si está instalado)
- Ranking por RMSE de test + AIC
- Guarda best_model, leaderboard, y gráfico comparativo
Salidas: outputs/model_selection/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------- Config --------
EDA_DIR = Path("outputs/eda")
OUT_DIR = Path("outputs/model_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_H = 52
SEASONAL_P = 52

def _yws_to_date(y, w):
    try:
        return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except Exception:
        return pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)

def cargar_serie():
    p = EDA_DIR / "casos_por_semana.csv"
    if not p.exists():
        raise FileNotFoundError("Falta outputs/eda/casos_por_semana.csv (ejecuta 01_EDA).")
    df = pd.read_csv(p)
    df.columns = [c.lower() for c in df.columns]
    fcol = next((c for c in df.columns if "fecha" in c), None)
    if fcol:
        df[fcol] = pd.to_datetime(df[fcol], errors="coerce")
        if df[fcol].notna().sum() > 10 and df[fcol].nunique() > 10:
            ycol = [c for c in df.columns if c != fcol][0]
            s = df.rename(columns={fcol: "fecha_sem", ycol: "casos"})[["fecha_sem", "casos"]]
        else:
            fcol = None
    if not fcol:
        acol = next((c for c in ["anio","año","ano","year"] if c in df.columns), None)
        scol = next((c for c in ["semana","semana_epidemiologica","semanaestadistica","nsemana"] if c in df.columns), None)
        tcol = next((c for c in df.columns if c not in [acol, scol]), None)
        s = df[[acol, scol, tcol]].dropna().copy()
        s["fecha_sem"] = s.apply(lambda r: _yws_to_date(r[acol], r[scol]), axis=1)
        s = s.rename(columns={tcol: "casos"})[["fecha_sem","casos"]]
    s = s.sort_values("fecha_sem")
    y = (s.set_index("fecha_sem")["casos"].resample("W-MON").sum()
           .asfreq("W-MON").interpolate())
    y = pd.to_numeric(y, errors="coerce").fillna(0)
    y[y<0]=0
    return y

def naive52(y, h):
    return y.shift(SEASONAL_P).iloc[-h:]

def fit_sarimax(y_tr, order, sorder):
    try:
        model = SARIMAX(y_tr, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        return res, None
    except Exception as e:
        return None, str(e)

def eval_candidate(y, order, sorder, name):
    # split
    h = TEST_H if len(y) > TEST_H + SEASONAL_P else max(6, len(y)//4)
    y_tr, y_te = y.iloc[:-h], y.iloc[-h:]
    res, err = fit_sarimax(y_tr, order, sorder)
    if res is None:
        return {"model": name, "order": str(order), "seasonal": str(sorder),
                "AIC": np.nan, "MAE": np.nan, "RMSE": np.nan, "error": err}
    pred = res.get_forecast(steps=len(y_te)).predicted_mean
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    return {"model": name, "order": str(order), "seasonal": str(sorder),
            "AIC": res.aic, "MAE": mae, "RMSE": rmse, "error": ""}, res, (y_tr, y_te, pred)

def try_prophet(y):
    try:
        from prophet import Prophet
    except Exception as e:
        return {"model": "Prophet", "order": "-", "seasonal": "weekly=TRUE",
                "AIC": np.nan, "MAE": np.nan, "RMSE": np.nan, "error": f"Prophet no disponible: {e}"}
    # preparar
    df = pd.DataFrame({"ds": y.index, "y": y.values})
    h = TEST_H if len(y) > TEST_H + SEASONAL_P else max(6, len(y)//4)
    df_tr, df_te = df.iloc[:-h], df.iloc[-h:]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(df_tr)
    fcst = m.predict(df_te[["ds"]])
    y_pred = pd.Series(fcst["yhat"].values, index=df_te["ds"].values)
    mae = mean_absolute_error(df_te["y"], y_pred)
    rmse = np.sqrt(mean_squared_error(df_te["y"], y_pred))
    return {"model": "Prophet", "order": "-", "seasonal": "weekly+yearly",
            "AIC": np.nan, "MAE": mae, "RMSE": rmse, "error": ""}

def main():
    print("[1/3] Cargando serie…")
    y = cargar_serie()
    print(f"Serie: {y.index.min().date()} → {y.index.max().date()} | n={len(y)}")

    print("[2/3] Grid SARIMAX + Naïve + Prophet(opc)")
    grid = [
        ((1,1,1), (1,1,0,SEASONAL_P)),
        ((1,1,2), (1,1,0,SEASONAL_P)),
        ((2,1,1), (1,1,0,SEASONAL_P)),
        ((1,1,1), (0,1,1,SEASONAL_P)),
        ((1,1,2), (0,1,1,SEASONAL_P)),
        ((2,1,1), (0,1,1,SEASONAL_P)),
        ((1,1,1), (1,1,1,SEASONAL_P)),
    ]

    rows, results_for_plot = [], []
    # Naïve 52
    h = TEST_H if len(y) > TEST_H + SEASONAL_P else max(6, len(y)//4)
    y_tr, y_te = y.iloc[:-h], y.iloc[-h:]
    y_nv = naive52(y, h)
    mae_nv = mean_absolute_error(y_te, y_nv) if y_nv.notna().sum()==len(y_nv) else np.nan
    rmse_nv = np.sqrt(mean_squared_error(y_te, y_nv)) if y_nv.notna().sum()==len(y_nv) else np.nan
    rows.append({"model":"Naive52","order":"-","seasonal":"-","AIC":np.nan,"MAE":mae_nv,"RMSE":rmse_nv,"error":""})

    # Grid SARIMAX
    for order, sorder in grid:
        rec, res, trio = eval_candidate(y, order, sorder, "SARIMAX")
        rows.append(rec)
        if rec["error"]=="":
            results_for_plot.append((trio[0], trio[1], trio[2], f"SARIMAX{order}{sorder}"))

    # Prophet opcional
    rows.append(try_prophet(y))

    lb = pd.DataFrame(rows).sort_values(["RMSE","AIC"], na_position="last")
    lb.to_csv(OUT_DIR / "leaderboard.csv", index=False)
    print(lb.head(10))

    # Mejor modelo (primera fila no Naive ni Prophet con RMSE válido)
    best = lb[lb["model"]=="SARIMAX"].dropna(subset=["RMSE"]).head(1)
    if best.empty:
        print("[!] No hubo modelo SARIMAX válido. Revisa leaderboard.csv")
        return
    best_order = eval(best["order"].values[0])
    best_sorder = eval(best["seasonal"].values[0])
    print(f"[✓] Mejor: SARIMAX{best_order}{best_sorder}")

    # Reentrenar en train y pronosticar test + futuro 26w
    res, _ = fit_sarimax(y_tr, best_order, best_sorder)
    pred_test = res.get_forecast(steps=len(y_te))
    y_pred = pred_test.predicted_mean
    ci_test = pred_test.conf_int(alpha=0.05); ci_test.columns=["lower","upper"]

    fut = res.get_forecast(steps=len(y_te)+26)
    fut_mean = fut.predicted_mean.iloc[-26:]
    fut_ci = fut.conf_int(alpha=0.05).iloc[-26:]; fut_ci.columns=["lower","upper"]

    # Guardar best_model y forecast
    try:
        import joblib
        Path("outputs/forecast").mkdir(parents=True, exist_ok=True)
        joblib.dump(res, OUT_DIR / "best_model.pkl")
    except Exception as e:
        print(f"[i] No se guardó el modelo: {e}")

    pd.DataFrame({
        "fecha": fut_mean.index, "yhat": fut_mean.values,
        "yhat_lower": fut_ci["lower"].values, "yhat_upper": fut_ci["upper"].values
    }).to_csv(OUT_DIR / "best_forecast_26w.csv", index=False)

    # Gráfico comparativo simple (mejor modelo)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12,5))
    plt.plot(y_tr.index, y_tr.values, label="Train", color="royalblue", lw=1.5)
    plt.plot(y_te.index, y_te.values, label="Real (Test)", color="darkgreen", lw=2)
    plt.plot(y_pred.index, y_pred.values, label="Pred Test (best)", color="darkorange", lw=2, ls="--")
    plt.fill_between(ci_test.index, ci_test["lower"], ci_test["upper"], color="orange", alpha=0.2, label="IC95% Test")
    plt.plot(fut_mean.index, fut_mean.values, label="Forecast 26w", color="red", lw=2, ls="--")
    plt.fill_between(fut_ci.index, fut_ci["lower"], fut_ci["upper"], color="red", alpha=0.15, label="IC95% Forecast")
    plt.title(f"Selección de Modelo — Mejor SARIMAX{best_order}{best_sorder}")
    plt.xlabel("Fecha"); plt.ylabel("Casos"); plt.legend(); plt.grid(alpha=.4, ls="--")
    plt.tight_layout(); plt.savefig(OUT_DIR / "best_model_plot.png", dpi=150); plt.close()

    print(f"\n✅ Listo. Revisa: {OUT_DIR}")

if __name__ == "__main__":
    main()
