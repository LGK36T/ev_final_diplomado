# -*- coding: utf-8 -*-
"""
ETL → Modelo Estrella para Urgencias Respiratorias
- Construye dimensiones y tabla de hechos desde el dataset original
- Grano de HechosUrgencias: semana (DimTiempo) × establecimiento × causa
- DimTemperatura: Incluye Ciudad/Comuna para distinguir zonas climáticas.
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
CLIMA_PATH = BASE / "data" / "clima_anio_semana.csv"

OUT_DIR = BASE / "outputs" / "modelo_datos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columnas esperadas en el dataset principal (nombres normalizados)
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
    "num5a14anios", "num15a64anios", "num65omas"
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
    """Limpia nombres de columnas: minúsculas, sin espacios, sin caracteres raros"""
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
    # ---------- Leer y normalizar Dataset Principal ----------
    df = load_any(IN_PATHS)
    df = normalize_cols(df)

    # Verificar columnas
    missing = sorted(list(REQ - set(df.columns)))
    if missing:
        print("[!] Faltan columnas en el dataset:", missing)

    # Convertir tipos numéricos clave
    for c in [
        "anio", "semanaestadistica", "ordencausa", "serviciosaludcodigo",
        "establecimientocodigo", "latitud", "longitud",
        "numtotal", "nummenor1anio", "num1a4anios",
        "num5a14anios", "num15a64anios", "num65omas"
    ]:
        if c in df.columns:
            df[c] = as_num(df[c])

    # ==========================================
    # 1. GENERACIÓN DE DIMENSIONES BÁSICAS
    # ==========================================

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

    # ==========================================
    # 2. DIMENSIÓN TIEMPO (CORREGIDA)
    # ==========================================
    if {"anio", "semanaestadistica"}.issubset(df.columns):
        dim_tiempo = (
            df[["anio", "semanaestadistica"]]
            .dropna()
            .drop_duplicates()
            .astype({"anio": "Int64", "semanaestadistica": "Int64"})
        )
        # ID Sintético (ej: 202301)
        dim_tiempo["ID_Tiempo"] = (
            dim_tiempo["anio"] * 100 + dim_tiempo["semanaestadistica"]
        )

        # Calcular fecha de inicio de semana para extraer mes
        fecha_inicio_semana = (
            pd.to_datetime(dim_tiempo["anio"].astype(str), errors="coerce")
            + pd.to_timedelta(
                (dim_tiempo["semanaestadistica"] - 1).clip(lower=0) * 7, unit="D"
            )
        )

        # --- CORRECCIÓN: Usar .dt accessor en lugar de PeriodIndex ---
        dim_tiempo["Mes"] = fecha_inicio_semana.dt.month.astype("Int64")
        
        dim_tiempo["Trimestre"] = (
            (dim_tiempo["Mes"] - 1) // 3 + 1
        ).astype("Int64")
        dim_tiempo["EsInvierno"] = dim_tiempo["Mes"].isin([6, 7, 8]).astype(int)

        dim_tiempo = dim_tiempo[
            ["ID_Tiempo", "anio", "semanaestadistica", "Mes", "Trimestre", "EsInvierno"]
        ].sort_values(["anio", "semanaestadistica"])
        
        dim_tiempo.to_csv(OUT_DIR / "DimTiempo.csv", index=False)
    else:
        dim_tiempo = pd.DataFrame()
        print("[!] No se pudo construir DimTiempo (faltan anio/semanaestadistica).")

    # ==========================================
    # 3. DIMENSIÓN ESTABLECIMIENTO
    # ==========================================
    estab_cols = [
        "establecimientocodigo", "establecimientoglosa", "regioncodigo",
        "comunacodigo", "serviciosaludcodigo", "tipoestablecimiento",
        "dependenciaadministrativa", "nivelatencion", "tipourgencia",
        "nivelcomplejidad", "latitud", "longitud"
    ]
    estab_cols = [c for c in estab_cols if c in df.columns]
    dim_estab = (
        df[estab_cols]
        .dropna(subset=["establecimientocodigo"])
        .drop_duplicates()
        .sort_values("establecimientocodigo")
    )
    dim_estab.to_csv(OUT_DIR / "DimEstablecimiento.csv", index=False)

    # ==========================================
    # 4. DIMENSIÓN TEMPERATURA (CON CIUDAD)
    # ==========================================
    dim_temp = pd.DataFrame()
    try:
        if CLIMA_PATH.exists():
            clima = pd.read_csv(CLIMA_PATH)
            clima = normalize_cols(clima) # Normaliza a minúsculas (Ciudad -> ciudad)

            # Buscamos 'ciudad' o 'comuna'
            col_ciudad = "ciudad" if "ciudad" in clima.columns else "comuna" if "comuna" in clima.columns else None
            
            needed = {"anio", "semanaestadistica", "temperatura"}
            if col_ciudad:
                needed.add(col_ciudad)

            if not needed.issubset(clima.columns) or not col_ciudad:
                print(
                    f"[!] DimTemperatura: Faltan columnas (se requiere anio, semana, temperatura y ciudad/comuna). "
                    f"Columnas actuales: {list(clima.columns)}"
                )
            else:
                # Renombramos la columna de ciudad para estandarizar
                clima.rename(columns={col_ciudad: "ciudad"}, inplace=True)

                # Filtrar nulos
                clima = clima.dropna(subset=["anio", "semanaestadistica", "ciudad"])
                
                # Tipos de datos
                clima["anio"] = as_num(clima["anio"]).astype("Int64")
                clima["semanaestadistica"] = as_num(clima["semanaestadistica"]).astype("Int64")
                clima["temperatura"] = as_num(clima["temperatura"])

                # --- AGREGAR POR CIUDAD ---
                # Promediamos si hay múltiples registros para la misma ciudad-semana
                clima = (
                    clima.groupby(["anio", "semanaestadistica", "ciudad"], as_index=False)[
                        "temperatura"
                    ]
                    .mean()
                )

                # Clave tiempo para cruces
                clima["ID_Tiempo"] = (
                    clima["anio"] * 100 + clima["semanaestadistica"]
                ).astype("Int64")

                # Filtrar solo semanas válidas
                if not dim_tiempo.empty:
                    clima = clima[
                        clima["ID_Tiempo"].isin(dim_tiempo["ID_Tiempo"])
                    ].copy()

                # Generar ID único para esta tabla (opcional)
                clima["TemperaturaID"] = range(1, len(clima) + 1)

                dim_temp = clima[
                    ["TemperaturaID", "ID_Tiempo", "anio", "semanaestadistica", "ciudad", "temperatura"]
                ].sort_values(["anio", "semanaestadistica", "ciudad"])

                dim_temp.to_csv(OUT_DIR / "DimTemperatura.csv", index=False)
                print(f"[i] DimTemperatura generada con {len(dim_temp)} filas.")
        else:
            print(f"[i] No se encontró archivo de clima en {CLIMA_PATH}")
    except Exception as e:
        print(f"[!] Error al construir DimTemperatura: {e}")
        dim_temp = pd.DataFrame()

    # ==========================================
    # 5. TABLA DE HECHOS
    # ==========================================
    fact_cols = [
        "establecimientocodigo", "ordencausa",
        "anio", "semanaestadistica",
        "numtotal", "nummenor1anio", "num1a4anios",
        "num5a14anios", "num15a64anios", "num65omas"
    ]
    fact_cols = [c for c in fact_cols if c in df.columns]
    fact = df[fact_cols].copy()

    # Generar FK tiempo
    if {"anio", "semanaestadistica"}.issubset(fact.columns):
        fact["ID_Tiempo"] = fact["anio"] * 100 + fact["semanaestadistica"]
    else:
        fact["ID_Tiempo"] = np.nan

    # Limpieza de Métricas
    metricas = [
        c for c in [
            "numtotal", "nummenor1anio", "num1a4anios",
            "num5a14anios", "num15a64anios", "num65omas"
        ] if c in fact.columns
    ]
    for m in metricas:
        fact[m] = as_num(fact[m]).fillna(0).astype(int)

    # Seleccionar y limpiar
    keep = ["ID_Tiempo", "establecimientocodigo", "ordencausa"] + metricas
    keep = [c for c in keep if c in fact.columns]
    
    fact = fact[keep].dropna(
        subset=["ID_Tiempo", "establecimientocodigo", "ordencausa"]
    )

    # Agrupar (Suma de métricas)
    fact = fact.groupby(
        ["ID_Tiempo", "establecimientocodigo", "ordencausa"],
        as_index=False
    )[metricas].sum()

    fact.to_csv(OUT_DIR / "HechosUrgencias.csv", index=False)

    # ==========================================
    # 6. CHECKS Y REPORTES
    # ==========================================
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

    # Verificar integridad referencial (FKs)
    if not dim_tiempo.empty and "ID_Tiempo" in fact.columns:
        fk_time_miss = fact.loc[~fact["ID_Tiempo"].isin(dim_tiempo["ID_Tiempo"])]
        if not fk_time_miss.empty:
            fk_time_miss.to_csv(OUT_DIR / "_WARN_FK_Tiempo_faltantes.csv", index=False)
            print("[!] Advertencia: filas en Hechos sin Tiempo válido -> _WARN_FK_Tiempo_faltantes.csv")

    if "establecimientocodigo" in fact.columns and "establecimientocodigo" in dim_estab.columns:
        miss_estab = fact.loc[~fact["establecimientocodigo"].isin(dim_estab["establecimientocodigo"])]
        if not miss_estab.empty:
            miss_estab.to_csv(OUT_DIR / "_WARN_FK_Estab_faltantes.csv", index=False)
            print("[!] Advertencia: filas en Hechos sin Establecimiento válido -> _WARN_FK_Estab_faltantes.csv")

    print(f"\n✅ Proceso finalizado. Archivos en: {OUT_DIR}")


if __name__ == "__main__":
    main()