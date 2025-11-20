# -*- coding: utf-8 -*-
"""
ETL → Modelo Estrella para Urgencias Respiratorias
- Construye dimensiones y tabla de hechos desde el dataset original
- Grano de HechosUrgencias: semana (DimTiempo) × establecimiento × causa
Salidas: outputs/modelo_datos/
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
BASE = Path(".")
IN_PATHS = [
    BASE / "outputs" / "eda" / "dataset_limpio.csv",      # preferido
    BASE / "data" / "parquetreader_Region de antofa.csv"  # alternativo
]
OUT_DIR = BASE / "outputs" / "modelo_datos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivo de clima semanal (ciudad, anio, semanaestadistica, temperatura)
CLIMA_PATH = BASE / "data" / "clima_anio_semana.csv"

# Columnas esperadas (nombres en minúsculas)
REQ = {
    "establecimientocodigo", "establecimientoglosa",
    "regioncodigo", "regionglosa",
    "comunacodigo", "comunaglosa",
    "serviciosaludcodigo", "serviciosaludglosa",
    "tipoestablecimiento", "dependenciaadministrativa", "nivelatencion",
    "tipourgencia", "nivelcomplejidad",
    "latitud", "longitud",
    "anio", "semanaestadistica",
    "ordencausa", "causa",
    "numtotal", "nummenor1anio", "num1a4anios",
    "num5a14anios", "num15a64anios", "num65omas",
}


def load_any(paths):
    last_err = None
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                print(f"[i] Cargado: {p}  shape={df.shape}")
                return df
            except Exception as e:
                last_err = e
    raise last_err or FileNotFoundError(
        "No se encontró dataset_limpio.csv ni el CSV original."
    )


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def as_num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    # ---------- Leer y normalizar ----------
    df = load_any(IN_PATHS)
    df = normalize_cols(df)

    # Verificar columnas
    missing = sorted(list(REQ - set(df.columns)))
    if missing:
        print("[!] Faltan columnas en el dataset:", missing)
        # seguimos con lo disponible para no cortar el flujo

    # Tipos
    for c in [
        "anio", "semanaestadistica", "ordencausa", "serviciosaludcodigo",
        "establecimientocodigo", "latitud", "longitud",
        "numtotal", "nummenor1anio", "num1a4anios",
        "num5a14anios", "num15a64anios", "num65omas",
    ]:
        if c in df.columns:
            df[c] = as_num(df[c])

    # ---------- Dimensiones ----------
    # DimRegion
    dim_region = (
        df[["regioncodigo", "regionglosa"]]
        .dropna(subset=["regioncodigo"]).drop_duplicates()
        .sort_values("regioncodigo")
    )
    dim_region.to_csv(OUT_DIR / "DimRegion.csv", index=False)

    # DimComuna
    dim_comuna = (
        df[["comunacodigo", "comunaglosa", "regioncodigo"]]
        .dropna(subset=["comunacodigo"]).drop_duplicates()
        .sort_values(["regioncodigo", "comunacodigo"])
    )
    dim_comuna.to_csv(OUT_DIR / "DimComuna.csv", index=False)

    # DimServicioSalud
    dim_ss = (
        df[["serviciosaludcodigo", "serviciosaludglosa"]]
        .dropna(subset=["serviciosaludcodigo"]).drop_duplicates()
        .sort_values("serviciosaludcodigo")
    )
    dim_ss.to_csv(OUT_DIR / "DimServicioSalud.csv", index=False)

    # DimCausa
    dim_causa = (
        df[["ordencausa", "causa"]]
        .dropna(subset=["ordencausa"]).drop_duplicates()
        .sort_values("ordencausa")
    )
    dim_causa.to_csv(OUT_DIR / "DimCausa.csv", index=False)

    # DimTiempo (ID_Tiempo sintético = anio*100 + semana)
    if {"anio", "semanaestadistica"}.issubset(df.columns):
        dim_tiempo = (
            df[["anio", "semanaestadistica"]]
            .dropna()
            .drop_duplicates()
            .astype({"anio": "Int64", "semanaestadistica": "Int64"})
        )
        dim_tiempo["ID_Tiempo"] = dim_tiempo["anio"] * 100 + dim_tiempo["semanaestadistica"]

        # Fecha aproximada de inicio de semana
        fecha_inicio_semana = (
            pd.to_datetime(dim_tiempo["anio"].astype(str), errors="coerce")
            + pd.to_timedelta((dim_tiempo["semanaestadistica"] - 1).clip(lower=0) * 7, unit="D")
        )

        dim_tiempo["Mes"] = fecha_inicio_semana.dt.month.astype("Int64")
        dim_tiempo["Trimestre"] = ((dim_tiempo["Mes"] - 1) // 3 + 1).astype("Int64")
        dim_tiempo["EsInvierno"] = dim_tiempo["Mes"].isin([6, 7, 8]).astype(int)

        dim_tiempo = (
            dim_tiempo[["ID_Tiempo", "anio", "semanaestadistica", "Mes", "Trimestre", "EsInvierno"]]
            .sort_values(["anio", "semanaestadistica"])
        )
        dim_tiempo.to_csv(OUT_DIR / "DimTiempo.csv", index=False)
    else:
        dim_tiempo = pd.DataFrame()
        print("[!] No se pudo construir DimTiempo (faltan anio/semanaestadistica).")

    # DimEstablecimiento
    estab_cols = [
        "establecimientocodigo", "establecimientoglosa", "regioncodigo", "comunacodigo",
        "serviciosaludcodigo", "tipoestablecimiento", "dependenciaadministrativa",
        "nivelatencion", "tipourgencia", "nivelcomplejidad", "latitud", "longitud",
    ]
    estab_cols = [c for c in estab_cols if c in df.columns]
    dim_estab = (
        df[estab_cols]
        .dropna(subset=["establecimientocodigo"])
        .drop_duplicates()
        .sort_values("establecimientocodigo")
    )
    dim_estab.to_csv(OUT_DIR / "DimEstablecimiento.csv", index=False)

    # ---------- DimTemperatura (con ciudad) ----------
    if CLIMA_PATH.exists():
        clima = pd.read_csv(CLIMA_PATH)
        clima = clima.rename(columns=str.lower)

        expected = {"ciudad", "anio", "semanaestadistica", "temperatura"}
        faltan_clima = expected - set(clima.columns)
        if faltan_clima:
            print("[!] Faltan columnas en clima_anio_semana.csv:", faltan_clima)
            dim_temp = pd.DataFrame()
        else:
            clima["anio"] = as_num(clima["anio"]).astype("Int64")
            clima["semanaestadistica"] = as_num(clima["semanaestadistica"]).astype("Int64")
            clima["temperatura"] = as_num(clima["temperatura"])

            # Clave alineada con DimTiempo
            clima["TemperaturaCodigo"] = clima["anio"] * 100 + clima["semanaestadistica"]

            dim_temp = (
                clima[["TemperaturaCodigo", "ciudad", "temperatura"]]
                .dropna(subset=["TemperaturaCodigo"])
                .drop_duplicates()
                .sort_values(["ciudad", "TemperaturaCodigo"])
            )
            dim_temp.to_csv(OUT_DIR / "DimTemperatura.csv", index=False)
    else:
        dim_temp = pd.DataFrame()
        print(f"[i] No se encontró archivo de clima en {CLIMA_PATH}; no se genera DimTemperatura.")

    # ---------- Tabla de Hechos ----------
    fact_cols = [
        "establecimientocodigo", "ordencausa",
        "anio", "semanaestadistica",
        "numtotal", "nummenor1anio", "num1a4anios",
        "num5a14anios", "num15a64anios", "num65omas",
    ]
    fact_cols = [c for c in fact_cols if c in df.columns]
    fact = df[fact_cols].copy()

    # Clave tiempo
    if {"anio", "semanaestadistica"}.issubset(fact.columns):
        fact["ID_Tiempo"] = fact["anio"] * 100 + fact["semanaestadistica"]
    else:
        fact["ID_Tiempo"] = np.nan

    # Orden y tipos
    metricas = [
        c for c in [
            "numtotal", "nummenor1anio", "num1a4anios",
            "num5a14anios", "num15a64anios", "num65omas",
        ]
        if c in fact.columns
    ]
    for m in metricas:
        fact[m] = as_num(fact[m]).fillna(0).astype(int)

    keep = ["ID_Tiempo", "establecimientocodigo", "ordencausa"] + metricas
    keep = [c for c in keep if c in fact.columns]
    fact = fact[keep].dropna(subset=["ID_Tiempo", "establecimientocodigo", "ordencausa"])

    # Agrupar y sumar para asegurar grano único
    fact = fact.groupby(
        ["ID_Tiempo", "establecimientocodigo", "ordencausa"],
        as_index=False
    )[metricas].sum()

    fact.to_csv(OUT_DIR / "HechosUrgencias.csv", index=False)

    # ---------- Checks rápidos ----------
    print("\n[ Resumen exportación ]")
    for name, df_ in [
        ("DimRegion", dim_region),
        ("DimComuna", dim_comuna),
        ("DimServicioSalud", dim_ss),
        ("DimCausa", dim_causa),
        ("DimTiempo", dim_tiempo),
        ("DimEstablecimiento", dim_estab),
        ("DimTemperatura", dim_temp),
        ("HechosUrgencias", fact),
    ]:
        if not df_.empty:
            print(f"  - {name:20s}: {df_.shape}")
        else:
            print(f"  - {name:20s}: (Vacío o no generado)")

    # FK faltantes vs DimTiempo
    if not dim_tiempo.empty and "ID_Tiempo" in fact.columns:
        fk_time_miss = fact.loc[~fact["ID_Tiempo"].isin(dim_tiempo["ID_Tiempo"])]
        if not fk_time_miss.empty:
            fk_time_miss.to_csv(
                OUT_DIR / "_WARN_FK_Tiempo_faltantes.csv", index=False
            )
            print(
                "[!] Advertencia: hay filas en hechos sin match en DimTiempo → "
                "_WARN_FK_Tiempo_faltantes.csv"
            )

    if "establecimientocodigo" in fact.columns and "establecimientocodigo" in dim_estab.columns:
        miss_estab = fact.loc[
            ~fact["establecimientocodigo"].isin(dim_estab["establecimientocodigo"])
        ]
        if not miss_estab.empty:
            miss_estab.to_csv(
                OUT_DIR / "_WARN_FK_Estab_faltantes.csv", index=False
            )
            print(
                "[!] Advertencia: hay filas en hechos sin match en DimEstablecimiento → "
                "_WARN_FK_Estab_faltantes.csv"
            )

    if "ordencausa" in fact.columns and "ordencausa" in dim_causa.columns:
        miss_causa = fact.loc[
            ~fact["ordencausa"].isin(dim_causa["ordencausa"])
        ]
        if not miss_causa.empty:
            miss_causa.to_csv(
                OUT_DIR / "_WARN_FK_Causa_faltantes.csv", index=False
            )
            print(
                "[!] Advertencia: hay filas en hechos sin match en DimCausa → "
                "_WARN_FK_Causa_faltantes.csv"
            )

    # Check FK tiempo vs DimTemperatura (opcional)
    if not dim_temp.empty and "TemperaturaCodigo" in dim_temp.columns:
        miss_temp = dim_tiempo.loc[
            ~dim_tiempo["ID_Tiempo"].isin(dim_temp["TemperaturaCodigo"])
        ]
        if not miss_temp.empty:
            miss_temp.to_csv(
                OUT_DIR / "_WARN_FK_Tiempo_sin_temperatura.csv", index=False
            )
            print(
                "[!] Advertencia: hay tiempos sin match en DimTemperatura → "
                "_WARN_FK_Tiempo_sin_temperatura.csv"
            )

    print(f"\n✅ Listo. Archivos en: {OUT_DIR}")


if __name__ == "__main__":
    main()
