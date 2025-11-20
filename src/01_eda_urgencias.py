# -*- coding: utf-8 -*-
"""
Proyecto: Urgencias Respiratorias - Región de Antofagasta
Etapa 1: Exploración y análisis descriptivo de datos (robusto)

"""

import os
import re
from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======= CONFIG =======
# Usamos pathlib para un manejo de rutas más robusto
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "data" / "parquetreader_Region de antofa.csv"
OUT_DIR = BASE_DIR / "outputs" / "eda"
FIG_DIR = OUT_DIR / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======= CARGA ROBUSTA =======
try:
    import chardet  # opcional
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

def cargar_csv_robusto(path: str) -> pd.DataFrame:
    """
    Intenta leer el CSV detectando encoding y separador.
    Tolera líneas defectuosas y encabezados 'meta' (skiprows).
    """
    encodings_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    if HAS_CHARDET:
        try:
            with open(path, "rb") as f:
                raw = f.read(150_000)
            enc_guess = chardet.detect(raw)["encoding"]
            if enc_guess and enc_guess.lower() not in [e.lower() for e in encodings_try]:
                encodings_try = [enc_guess] + encodings_try
        except Exception:
            pass

    last_err = None
    # A) sep auto
    for enc in encodings_try:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="warn")
            print(f"[i] Leído con sep=auto, encoding={enc}. Shape={df.shape}")
            return df
        except Exception as e:
            last_err = e

    # B) sep explícitos
    for enc in encodings_try:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc, on_bad_lines="warn")
                print(f"[i] Leído con sep='{sep}', encoding={enc}. Shape={df.shape}")
                return df
            except Exception as e:
                last_err = e

    # C) encabezado desplazado (skiprows)
    for enc in encodings_try:
        for sep in [",", ";", "\t", "|"]:
            for skip in range(1, 10):
                try:
                    df = pd.read_csv(path, sep=sep, engine="python", encoding=enc, skiprows=skip, on_bad_lines="warn")
                    print(f"[i] Leído con sep='{sep}', encoding={enc}, skiprows={skip}. Shape={df.shape}")
                    return df
                except Exception as e:
                    last_err = e
    
    if last_err:
        raise last_err
    raise FileNotFoundError("No se pudo cargar el archivo CSV con ninguna configuración probada.")


# ======= UTILIDADES =======
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace(r"[^\w_]", "", regex=True))
    return df

def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = df.columns.tolist()
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return s

# ======= EJECUCIÓN =======
def main():
    print("🔹 Cargando dataset (robusto)...")
    df = cargar_csv_robusto(DATA_PATH)
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}\n")

    # ---- Limpieza básica
    print("🔹 Limpieza y normalización de columnas...")
    df = df.drop_duplicates()
    df = normalize_cols(df)

    # Resumen de faltantes
    faltantes = df.isna().mean().sort_values(ascending=False) * 100
    print("\n% faltantes (top 10):")
    print(faltantes.head(10))

    # Tipos de datos
    print("\nTipos de datos (conteo):")
    print(df.dtypes.astype(str).value_counts())

    # ---- Detección de columnas clave
    print("\n🔹 Detección automática de columnas clave...")

    col_comuna = first_existing(df, ["comunaglosa", "comuna", "comuna_nombre", "nom_comuna"]) or find_col(df, [r"comuna"])
    col_fecha = first_existing(df, ["fecha", "f_evento", "f_semana", "fecha_semana"])
    if not col_fecha:
        col_fecha = find_col(df, [r"fecha", r"^f_.*", r"semana", r"^anio$", r"^año$", r"^ano$", r"^year$"])
    col_semana = first_existing(df, ["semanaestadistica", "semana", "semana_epidemiologica", "nsemana"]) or find_col(df, [r"semana"])
    col_anio = first_existing(df, ["anio", "año", "ano", "year"]) or find_col(df, [r"^a[ñn]o$", r"^anio$", r"year"])
    col_estab = first_existing(df, ["establecimientoglosa", "establecimiento", "tipo_establecimiento"]) or find_col(df, [r"estable", r"sapu|sar|hospital|clinica|urgenc"])
    col_causa = first_existing(df, ["causa", "diagnostico", "causa_respiratoria", "categoria"]) or find_col(df, [r"causa", r"diagn", r"respir"])
    col_edad = first_existing(df, ["grupo_etario", "edad_grupo", "grupoedad", "grupo_etario_desc"]) or find_col(df, [r"edad", r"etario"])
    col_total = first_existing(df, ["numtotal", "total", "casos", "n_casos", "cantidad", "conteo"]) or find_col(df, [r"total|casos|cantidad|conteo"])

    print("\nColumnas identificadas:")
    print(f"📍 comuna: {col_comuna}")
    print(f"📅 fecha: {col_fecha} | semana: {col_semana} | año: {col_anio}")
    print(f"🏥 establecimiento: {col_estab}")
    print(f"🦠 causa: {col_causa}")
    print(f"👶 grupo etario: {col_edad}")
    print(f"📊 total: {col_total}")

    if not col_total:
        raise ValueError("No se pudo identificar la columna de total/casos. Revisa los encabezados.")
    if not (col_fecha or col_semana or col_anio):
        print("[!] No se identificó fecha/semana/año. El análisis temporal será limitado.")

    # ---- Casting de tipos
    if col_fecha:
        df[col_fecha] = coerce_datetime(df[col_fecha])
    if col_anio and not np.issubdtype(df[col_anio].dtype, np.number):
        df[col_anio] = pd.to_numeric(df[col_anio], errors="coerce").astype("Int64")
    if col_semana and not np.issubdtype(df[col_semana].dtype, np.number):
        df[col_semana] = pd.to_numeric(df[col_semana], errors="coerce").astype("Int64")

    # ---- Guardar copia limpia
    out_clean = OUT_DIR / "dataset_limpio.csv"
    df.to_csv(out_clean, index=False, encoding="utf-8")
    print(f"\n[✓] Copia limpia guardada en: {out_clean}")

    # ======= DESCRIPTIVO =======
    print("\n🔹 Generando resúmenes...")

    # 1) Casos por semana
    serie_sem = None
    if col_fecha and pd.api.types.is_datetime64_any_dtype(df[col_fecha]):
        serie_sem = df.groupby(col_fecha)[col_total].sum().reset_index()
        serie_sem = serie_sem.sort_values(col_fecha)
        serie_sem.to_csv(OUT_DIR / "casos_por_semana.csv", index=False)
        print("[✓] casos_por_semana.csv")
    elif col_semana and col_anio:
        serie_sem = (df.dropna(subset=[col_semana, col_anio])
                       .groupby([col_anio, col_semana])[col_total].sum()
                       .reset_index()
                       .sort_values([col_anio, col_semana]))
        serie_sem.to_csv(OUT_DIR / "casos_por_semana.csv", index=False)
        print("[✓] casos_por_semana.csv")
    else:
        print("[i] No se pudo crear serie semanal (faltan columnas temporales).")

    # 2) Casos por comuna
    resumen_comuna = None
    if col_comuna:
        resumen_comuna = (df.groupby(col_comuna)[col_total].sum()
                            .sort_values(ascending=False).reset_index())
        resumen_comuna.to_csv(OUT_DIR / "casos_por_comuna.csv", index=False)
        print("[✓] casos_por_comuna.csv")
    
    # 3) Casos por grupo etario
    resumen_edad = None
    if col_edad:
        resumen_edad = (df.groupby(col_edad)[col_total].sum()
                          .sort_values(ascending=False).reset_index())
        resumen_edad.to_csv(OUT_DIR / "casos_por_grupo_etario.csv", index=False)
        print("[✓] casos_por_grupo_etario.csv")

    # 4) Casos por establecimiento
    if col_estab:
        resumen_estab = (df.groupby(col_estab)[col_total].sum()
                           .sort_values(ascending=False).reset_index())
        resumen_estab.to_csv(OUT_DIR / "casos_por_establecimiento.csv", index=False)
        print("[✓] casos_por_establecimiento.csv")

    # 5) Causas respiratorias
    if col_causa:
        resumen_causas = (df.groupby(col_causa)[col_total].sum()
                            .sort_values(ascending=False).reset_index())
        resumen_causas.to_csv(OUT_DIR / "top_causas_respiratorias.csv", index=False)
        print("[✓] top_causas_respiratorias.csv")

    # ======= VISUALIZACIONES =======
    print("\n🔹 Generando visualizaciones...")
    sns.set_theme(style="whitegrid")

    # Serie temporal semanal
    if serie_sem is not None:
        plt.figure(figsize=(11, 4))
        x_col = serie_sem.columns[0]
        y_col = serie_sem.columns[1]
        if pd.api.types.is_datetime64_any_dtype(serie_sem[x_col]):
             plt.plot(serie_sem[x_col], serie_sem[y_col], lw=1.8, color="#1565C0")
        else: # Para el caso de año-semana
             # Creamos una etiqueta combinada para el eje x
             serie_sem['anio_semana'] = serie_sem[col_anio].astype(str) + "-S" + serie_sem[col_semana].astype(str).str.zfill(2)
             plt.plot(serie_sem['anio_semana'], serie_sem[col_total], lw=1.8, color="#1565C0")
             # Hacemos los ticks del eje x más legibles
             plt.xticks(rotation=90)
             ticks = plt.gca().get_xticks()
             labels = plt.gca().get_xticklabels()
             n = max(1, len(ticks) // 10) # Mostrar aprox. 10 etiquetas
             plt.xticks(ticks[::n], labels[::n])

        plt.title("Evolución semanal de urgencias respiratorias")
        plt.xlabel("Semana/Fecha"); plt.ylabel("Casos")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "serie_temporal_semanal.png", dpi=150)
        plt.close()
        print("[✓] Figura: serie_temporal_semanal.png")

    # Barras por comuna
    if resumen_comuna is not None:
        topc = resumen_comuna.head(20) # Mostramos solo las top 20 para legibilidad
        plt.figure(figsize=(10, max(4, min(12, len(topc) * 0.4))))
        sns.barplot(
            data=topc, 
            x=topc.columns[1], 
            y=topc.columns[0], 
            palette="viridis",
            hue=topc.columns[0], # <-- CORRECCIÓN AÑADIDA
            legend=False         # <-- CORRECCIÓN AÑADIDA
        )
        plt.title("Total de casos por comuna (Top 20)")
        plt.xlabel("Casos"); plt.ylabel("Comuna")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "casos_por_comuna.png", dpi=150)
        plt.close()
        print("[✓] Figura: casos_por_comuna.png")

    # Barras por grupo etario
    if resumen_edad is not None:
        e = resumen_edad.copy()
        plt.figure(figsize=(8, max(4, min(12, len(e) * 0.4))))
        sns.barplot(
            data=e, 
            x=e.columns[1], 
            y=e.columns[0], 
            palette="mako",
            hue=e.columns[0], # <-- CORRECCIÓN AÑADIDA
            legend=False      # <-- CORRECCIÓN AÑADIDA
        )
        plt.title("Distribución por grupo etario")
        plt.xlabel("Casos"); plt.ylabel("Grupo etario")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "casos_por_grupo_etario.png", dpi=150)
        plt.close()
        print("[✓] Figura: casos_por_grupo_etario.png")

    # Heatmap año-semana
    if col_anio and col_semana:
        try:
            heat = (df.dropna(subset=[col_anio, col_semana])
                      .groupby([col_anio, col_semana])[col_total].sum()
                      .unstack(fill_value=0))
            plt.figure(figsize=(12, max(4, 0.5 * len(heat.index))))
            sns.heatmap(heat, cmap="YlOrRd", linewidths=.5)
            plt.title("Heatmap anual de urgencias respiratorias")
            plt.xlabel("Semana Epidemiológica"); plt.ylabel("Año")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "heatmap_anual.png", dpi=150)
            plt.close()
            print("[✓] Figura: heatmap_anual.png")
        except Exception as e:
            print(f"[!] No se pudo generar el heatmap: {e}")


    print("\n✅ EDA completado. Revisa:", OUT_DIR)

if __name__ == "__main__":
    main()