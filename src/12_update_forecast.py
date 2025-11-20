# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

EDA_DIR = Path("outputs/eda")
SEL_DIR = Path("outputs/model_selection")
FRC_DIR = Path("outputs/forecast")
FRC_DIR.mkdir(parents=True, exist_ok=True)
TEST_H = 52

def _yws_to_date(y, w):
    try:
        return pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except Exception:
        return pd.Timestamp(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)

def load_series():
    p = EDA_DIR / "casos_por_semana.csv"
    df = pd.read_csv(p); df.columns = [c.lower() for c in df.columns]
    f = next((c for c in df.columns if "fecha" in c), None)
    if f:
        df[f] = pd.to_datetime(df[f], errors="coerce")
        ycol = [c for c in df.columns if c != f][0]
        s = df.rename(columns={f:"fecha_sem", ycol:"casos"})[["fecha_sem","casos"]]
    else:
        a = next((c for c in ["anio","año","ano","year"] if c in df.columns), None)
        w = next((c for c in ["semana","semana_epidemiologica","semanaestadistica","nsemana"] if c in df.columns), None)
        t = next((c for c in df.columns if c not in [a,w]), None)
        s = df[[a,w,t]].dropna().copy()
        s["fecha_sem"]=s.apply(lambda r:_yws_to_date(r[a],r[w]),axis=1)
        s = s.rename(columns={t:"casos"})[["fecha_sem","casos"]]
    s = s.sort_values("fecha_sem")
    y = (s.set_index("fecha_sem")["casos"].resample("W-MON").sum().asfreq("W-MON").interpolate())
    y = pd.to_numeric(y, errors="coerce").fillna(0); y[y<0]=0
    return y

def best_orders():
    lb = pd.read_csv(SEL_DIR / "leaderboard.csv")
    best = lb[lb["model"]=="SARIMAX"].dropna(subset=["RMSE"]).head(1)
    if best.empty:  # fallback
        return (1,1,1), (1,1,0,52)
    return eval(best["order"].values[0]), eval(best["seasonal"].values[0])

def main():
    y = load_series()
    order, sorder = best_orders()
    h = TEST_H if len(y) > TEST_H+52 else max(6, len(y)//4)
    y_tr, y_te = y.iloc[:-h], y.iloc[-h:]

    model = SARIMAX(y_tr, order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred = model.get_forecast(steps=len(y_te)).predicted_mean
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))

    # Exportar forecast futuro 26w
    fut = model.get_forecast(steps=len(y_te)+26).predicted_mean.iloc[-26:]
    fut.to_csv(FRC_DIR / "monthly_update_forecast_26w.csv")

    # Log de monitoreo
    logp = FRC_DIR / "monitoring_log.csv"
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "order": str(order), "seasonal": str(sorder),
        "MAE": mae, "RMSE": rmse, "n_points": len(y)
    }])
    if logp.exists():
        pd.concat([pd.read_csv(logp), row], ignore_index=True).to_csv(logp, index=False)
    else:
        row.to_csv(logp, index=False)

    print("✅ Actualización mensual lista. Ver:", FRC_DIR)

if __name__ == "__main__":
    main()
