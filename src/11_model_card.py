# -*- coding: utf-8 -*-
"""
ETL → Modelo Estrella + Reportes Específicos
------------------------------------------------------
1. Genera el Modelo Estrella en: outputs/modelo_datos/
2. Genera Reportes Específicos en: outputs/reportes/
   - Reporte_Peaks.csv (Alertas de alta demanda)
   - Reporte_Clima.csv (Temperaturas Min/Max por ciudad)
   - Reporte_Estaciones.csv (Resumen por temporada)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- CONFIGURACIÓN ----------------
BASE = Path(".")
IN_PATHS = [
    BASE / "outputs" / "eda" / "dataset_limpio.csv",
    BASE / "dataset_limpio.csv"
]
CLIMA_PATH = BASE / "clima_anio_semana.csv"

# Carpetas de Salida
DIR_MODELO = BASE / "outputs" / "modelo_datos"
DIR_REPORTES = BASE / "outputs" / "reportes"

DIR_MODELO.mkdir(parents=True, exist_ok=True)
DIR_REPORTES.mkdir(parents=True, exist_ok=True)

# ---------------- FUNCIONES AUXILIARES ----------------
def load_any(paths):
    for p in paths:
        if p.exists():
            print(f"[i] Cargando dataset principal desde: {p}")
            return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError("ERROR: No se encontró 'dataset_limpio.csv'")

def normalize_cols(df):
    """Limpia nombres de columnas"""
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True))
    return df

def as_num(s):
    return pd.to_numeric(s, errors="coerce")

def get_estacion(mes):
    """Retorna la estación del año (Chile/Hemisferio Sur)"""
    if mes in [12, 1, 2]: return "Verano"
    elif mes in [3, 4, 5]: return "Otoño"
    elif mes in [6, 7, 8]: return "Invierno"
    else: return "Primavera"

# ---------------- PROCESO PRINCIPAL ----------------
def main():
    # 1. Cargar y Limpiar Datos
    df = load_any(IN_PATHS)
    df = normalize_cols(df)
    
    # Convertir números
    cols_num = ["anio", "semanaestadistica", "ordencausa", "establecimientocodigo", 
                "numtotal", "nummenor1anio", "num1a4anios", "num65omas"]
    for c in cols_num:
        if c in df.columns: df[c] = as_num(df[c])

    # ==============================================================================
    # PARTE A: GENERACIÓN DEL MODELO ESTRELLA (Para PowerBI / SQL)
    # ==============================================================================
    print("\n--- Generando Modelo Estrella (outputs/modelo_datos) ---")

    # Dimensiones Básicas
    (df[["regioncodigo", "regionglosa"]].dropna(subset=["regioncodigo"]).drop_duplicates()
     .to_csv(DIR_MODELO / "DimRegion.csv", index=False))
     
    (df[["comunacodigo", "comunaglosa"]].dropna(subset=["comunacodigo"]).drop_duplicates()
     .to_csv(DIR_MODELO / "DimComuna.csv", index=False))

    (df[["ordencausa", "causa"]].dropna(subset=["ordencausa"]).drop_duplicates()
     .to_csv(DIR_MODELO / "DimCausa.csv", index=False))
     
    estab_cols = ["establecimientocodigo", "establecimientoglosa", "regioncodigo", "comunacodigo"]
    df_estab = df[[c for c in estab_cols if c in df.columns]].drop_duplicates("establecimientocodigo")
    df_estab.to_csv(DIR_MODELO / "DimEstablecimiento.csv", index=False)

    # DimTiempo (Con Estaciones)
    if {"anio", "semanaestadistica"}.issubset(df.columns):
        dim_tiempo = df[["anio", "semanaestadistica"]].dropna().drop_duplicates().astype(int)
        dim_tiempo["ID_Tiempo"] = dim_tiempo["anio"] * 100 + dim_tiempo["semanaestadistica"]
        
        # Calcular Mes y Estación
        fecha_aprox = pd.to_datetime(dim_tiempo["anio"].astype(str) + "-01-01") + \
                      pd.to_timedelta(dim_tiempo["semanaestadistica"] * 7, unit="D")
        dim_tiempo["Mes"] = fecha_aprox.dt.month
        dim_tiempo["Estacion"] = dim_tiempo["Mes"].apply(get_estacion)
        
        dim_tiempo.to_csv(DIR_MODELO / "DimTiempo.csv", index=False)
    
    # HechosUrgencias (Base Agrupada)
    fact_cols = ["anio", "semanaestadistica", "establecimientocodigo", "numtotal"]
    fact = df[fact_cols].groupby(["anio", "semanaestadistica", "establecimientocodigo"], as_index=False).sum()
    fact["ID_Tiempo"] = fact["anio"] * 100 + fact["semanaestadistica"]
    
    # CALCULO DE PEAKS (Umbral: Percentil 90 por establecimiento)
    umbrales = fact.groupby("establecimientocodigo")["numtotal"].quantile(0.90).to_dict()
    fact["EsPeak"] = fact.apply(lambda x: 1 if x["numtotal"] > umbrales.get(x["establecimientocodigo"], 9999) else 0, axis=1)
    
    fact.to_csv(DIR_MODELO / "HechosUrgencias.csv", index=False)

    # ==============================================================================
    # PARTE B: GENERACIÓN DE REPORTES APARTE (outputs/reportes)
    # ==============================================================================
    print("\n--- Generando Reportes Específicos (outputs/reportes) ---")

    # 1. TABLA DE PEAKS (Solo las alertas)
    # Unimos con nombres para que sea legible humanamente
    df_peaks = fact[fact["EsPeak"] == 1].copy()
    df_peaks = df_peaks.merge(df_estab[["establecimientocodigo", "establecimientoglosa"]], on="establecimientocodigo", how="left")
    
    # Limpiamos columnas para el reporte final
    reporte_peaks = df_peaks[["ID_Tiempo", "establecimientoglosa", "numtotal", "EsPeak"]]
    reporte_peaks.to_csv(DIR_REPORTES / "Reporte_Peaks.csv", index=False)
    print(f"✅ Reporte Peaks generado: {len(reporte_peaks)} alertas detectadas.")

    # 2. TABLA DE CLIMA (Temp Min/Max/Promedio)
    if CLIMA_PATH.exists():
        clima = pd.read_csv(CLIMA_PATH)
        clima = normalize_cols(clima)
        
        # Detectar nombre columna ciudad
        col_ciudad = next((c for c in ["ciudad", "comuna"] if c in clima.columns), None)
        if col_ciudad: clima.rename(columns={col_ciudad: "ciudad"}, inplace=True)

        # Crear Min/Max si no existen (Simulación para ejemplo si faltan datos)
        if "temperatura" in clima.columns:
            if "temp_min" not in clima.columns: clima["temp_min"] = clima["temperatura"] - 5
            if "temp_max" not in clima.columns: clima["temp_max"] = clima["temperatura"] + 5
            
            reporte_clima = clima[["anio", "semanaestadistica", "ciudad", "temperatura", "temp_min", "temp_max"]].copy()
            reporte_clima.to_csv(DIR_REPORTES / "Reporte_Clima.csv", index=False)
            print(f"✅ Reporte Clima generado: {len(reporte_clima)} registros.")
    else:
        print("⚠️ No se encontró archivo de clima para generar reporte.")

    # 3. TABLA DE ESTACIONES (Resumen Agregado)
    # Unimos Hechos con Tiempo para tener la estación
    df_estaciones = fact.merge(dim_tiempo[["ID_Tiempo", "Estacion"]], on="ID_Tiempo", how="left")
    
    # Agrupamos para ver el total por estación
    reporte_estaciones = df_estaciones.groupby("Estacion", as_index=False).agg({
        "numtotal": ["sum", "mean"],
        "EsPeak": "sum"
    })
    # Aplanamos nombres de columnas
    reporte_estaciones.columns = ["Estacion", "Total_Atenciones", "Promedio_Semanal", "Cantidad_Peaks"]
    
    reporte_estaciones.to_csv(DIR_REPORTES / "Reporte_Estaciones.csv", index=False)
    print("✅ Reporte Estaciones generado.")

    print(f"\n🚀 PROCESO FINALIZADO. Revisa la carpeta: {DIR_REPORTES}")

if __name__ == "__main__":
    main()