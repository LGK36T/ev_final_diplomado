# -*- coding: utf-8 -*-
"""
Análisis de Estacionalidad: Detecta Picos y su Temperatura Asociada
-------------------------------------------------------------------
1. Carga casos y clima.
2. Cruza los datos por Año-Semana-Ciudad.
3. Encuentra la semana con más casos (Peak).
4. Extrae la temperatura exacta de esa semana.
5. Guarda en: outputs/resumen/picos_casos_temperatura.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- Configuración ---
BASE_DIR = Path(".")
if (BASE_DIR / "outputs").exists():
    OUT_BASE = BASE_DIR / "outputs"
else:
    OUT_BASE = BASE_DIR.parent / "outputs"

DATA_PATH = OUT_BASE / "eda" / "dataset_limpio.csv"

# Búsqueda flexible del clima
CLIMA_PATH = BASE_DIR / "data" / "clima_anio_semana.csv"
if not CLIMA_PATH.exists():
    CLIMA_PATH = BASE_DIR / "clima_anio_semana.csv"
    if not CLIMA_PATH.exists():
        CLIMA_PATH = BASE_DIR.parent / "clima_anio_semana.csv"

RESUMEN_DIR = OUT_BASE / "resumen"
RESUMEN_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def load_data():
    """Carga casos y agrega la temperatura correspondiente"""
    print("--- Cargando datos ---")
    
    # 1. Cargar Casos
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    
    # Agrupar por semana y ciudad
    # (Normalizamos nombre de ciudad a minúsculas para evitar errores de cruce)
    casos = df.groupby(["anio", "semanaestadistica", "comunaglosa"])["numtotal"].sum().reset_index()
    casos.rename(columns={"comunaglosa": "ciudad", "numtotal": "casos"}, inplace=True)
    casos["ciudad"] = casos["ciudad"].astype(str).str.lower().str.strip()
    
    print(f"[i] Casos cargados. Ciudades encontradas: {casos['ciudad'].unique()}")

    # 2. Cargar Clima
    if CLIMA_PATH.exists():
        clima = pd.read_csv(CLIMA_PATH)
        clima.columns = clima.columns.str.lower().str.strip()
        
        # Detectar columna ciudad
        col_ciudad = "ciudad" if "ciudad" in clima.columns else "comuna"
        if col_ciudad in clima.columns:
            clima.rename(columns={col_ciudad: "ciudad"}, inplace=True)
            clima["ciudad"] = clima["ciudad"].astype(str).str.lower().str.strip()
            
            # Normalizar columnas de cruce
            clima["anio"] = pd.to_numeric(clima["anio"], errors='coerce')
            clima["semanaestadistica"] = pd.to_numeric(clima["semanaestadistica"], errors='coerce')
            
            # Cruce (Merge): A cada fila de casos, le pegamos su temperatura
            merged = pd.merge(casos, clima, on=["anio", "semanaestadistica", "ciudad"], how="left")
            print("[ok] Clima cruzado correctamente.")
            return merged
        else:
            print("[!] El archivo de clima no tiene columna 'ciudad' o 'comuna'.")
    else:
        print("[!] No se encontró archivo de clima. Las temperaturas saldrán vacías.")
    
    # Si falla el clima, devolvemos casos con NaN
    casos["temperatura"] = np.nan
    return casos

def detect_peaks(df):
    """Encuentra los máximos históricos y guarda la temperatura de ese momento"""
    resumen = []
    
    for ciudad in df["ciudad"].unique():
        subset = df[df["ciudad"] == ciudad].copy()
        if subset.empty: continue
        
        # --- A. Detección de Peak de CASOS ---
        idx_max_casos = subset["casos"].idxmax()
        row_casos = subset.loc[idx_max_casos]
        
        resumen.append({
            "ciudad": ciudad.title(), # Poner mayúscula bonita
            "tipo": "casos",          # Etiqueta para el reporte
            "anio": int(row_casos["anio"]),
            "semana": int(row_casos["semanaestadistica"]),
            "fecha": f"{int(row_casos['anio'])}-W{int(row_casos['semanaestadistica']):02d}",
            "valor": row_casos["casos"],        # Cantidad de enfermos
            "temp_en_semana": row_casos["temperatura"], # <--- LA TEMPERATURA ASOCIADA
            "casos_en_semana": row_casos["casos"]
        })
        
        # --- B. Detección de Peak de FRÍO (Temp Mínima) ---
        # (Solo si hay datos de temperatura)
        if subset["temperatura"].notna().any():
            idx_min_temp = subset["temperatura"].idxmin() # Buscamos el mínimo (frío)
            row_temp = subset.loc[idx_min_temp]
            
            resumen.append({
                "ciudad": ciudad.title(),
                "tipo": "temperatura_min",
                "anio": int(row_temp["anio"]),
                "semana": int(row_temp["semanaestadistica"]),
                "fecha": f"{int(row_temp['anio'])}-W{int(row_temp['semanaestadistica']):02d}",
                "valor": row_temp["temperatura"],    # Grados de temperatura
                "temp_en_semana": row_temp["temperatura"],
                "casos_en_semana": row_temp["casos"] # <--- CASOS ASOCIADOS AL FRÍO
            })

    return pd.DataFrame(resumen)

# --- Main ---
def main():
    print("\n=== Buscando Picos y Temperaturas Asociadas ===")
    try:
        df = load_data()
        
        picos_df = detect_peaks(df)
        
        # Guardar
        out_path = RESUMEN_DIR / "picos_casos_temperatura.csv"
        picos_df.to_csv(out_path, index=False)
        
        # Mostrar Resumen en Consola
        print("\n[RESUMEN DETECTADO]")
        print(f"{'CIUDAD':<15} | {'TIPO':<15} | {'FECHA':<10} | {'VALOR':<8} | {'TEMP ASOCIADA'}")
        print("-" * 70)
        
        for _, row in picos_df.iterrows():
            temp_str = f"{row['temp_en_semana']:.1f}°C" if pd.notna(row['temp_en_semana']) else "N/D"
            print(f"{row['ciudad']:<15} | {row['tipo']:<15} | {row['fecha']:<10} | {row['valor']:<8.1f} | {temp_str}")

        print(f"\n✅ Archivo guardado en: {out_path}")
        print("   (Ahora puedes ejecutar 04_resumen_estacionalidad.py para ver esto en el reporte)")
        
    except Exception as e:
        print(f"\n[ERROR CRITICO] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()