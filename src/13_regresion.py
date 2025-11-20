"""
Proyecto: Urgencias Respiratorias - Región de Antofagasta
Etapa 7: Análisis de Regresión Simple (N° Casos vs. Semana)
Basado en la imagen de referencia y el informe de estacionalidad (02)
"""

# ======= 1. BIBLIOTECAS (Imports) =======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ======= 2. VARIABLES GLOBALES (Constantes) =======
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "eda" / "dataset_limpio.csv" 
OUT_DIR = BASE_DIR / "outputs" / "modelo_regresion_simple"
FIG_DIR = OUT_DIR / "figuras"

# Columnas clave para este análisis
TARGET_COLUMN = 'numtotal'
TIME_COLUMN = 'semanaestadistica'

# ======= 3. FUNCIONES =======

def cargar_datos_limpios(path: Path) -> pd.DataFrame:
    """Carga el dataset limpio (generado por 01_eda)."""
    print(f"🔹 1. Cargando dataset limpio desde: {path.name}")
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-8-sig', on_bad_lines='warn')
        if df.shape[1] <= 1:
             df = pd.read_csv(path, sep=',', encoding='utf-8-sig', on_bad_lines='warn')
        
        if df.shape[1] <= 1:
             raise ValueError("No se pudo determinar el separador (ni ';' ni ',')")

        print(f"[i] Leído exitosamente. Shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] al cargar {path}: {e}")
        exit()

def agregar_datos_por_semana(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa el total de casos por semana estadística para 
    replicar el gráfico de referencia.
    """
    print("🔹 2. Agrupando casos totales por semana...")
    try:
        # Convertir a numérico por si acaso
        df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors='coerce')
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
        
        # Agrupar por semana y sumar todos los casos
        df_agregado = df.groupby(TIME_COLUMN)[TARGET_COLUMN].sum().reset_index()
        
        print(f"[i] Datos agregados. Shape={df_agregado.shape}")
        return df_agregado
        
    except Exception as e:
        print(f"[ERROR] al agregar datos: {e}")
        return pd.DataFrame()

def generar_plot_regresion_simple(df_agregado: pd.DataFrame, output_path: Path):
    """
    Genera el gráfico de Regresión Lineal Simple (como la imagen de referencia).
    """
    print("🔹 3. Generando Gráfico 1: Regresión Lineal Simple (Tendencia)...")
    plt.figure(figsize=(10, 6))
    
    # Usamos regplot para obtener la línea de regresión automáticamente
    sns.regplot(
        data=df_agregado,
        x=TIME_COLUMN,
        y=TARGET_COLUMN,
        scatter_kws={'alpha': 0.5, 's': 20}, # Puntos de datos
        line_kws={'color': 'red', 'linestyle': '--'} # Línea de regresión
    )
    
    plt.title('Regresión Lineal Simple: N° Casos vs Semana Estadística (Tendencia General)')
    plt.xlabel('Semana Estadística')
    plt.ylabel('N° Casos (Totales)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"[✓] Figura guardada en: {output_path}")
    plt.show()

def generar_plot_estacionalidad_real(df_agregado: pd.DataFrame, output_path: Path):
    """
    Genera el gráfico de línea que muestra la estacionalidad real
    (como se vio en el informe 02_estacionalidad.py).
    """
    print("🔹 4. Generando Gráfico 2: Relación Real (Estacionalidad)...")
    plt.figure(figsize=(10, 6))
    
    # Usamos lineplot para conectar los puntos y ver la curva
    sns.lineplot(
        data=df_agregado,
        x=TIME_COLUMN,
        y=TARGET_COLUMN,
        marker='o',
        color='darkblue'
    )
    
    plt.title('Relación Real (Estacionalidad): N° Casos vs Semana Estadística')
    plt.xlabel('Semana Estadística')
    plt.ylabel('N° Casos (Totales)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"[✓] Figura guardada en: {output_path}")
    plt.show()


# ======= 4. FUNCIÓN PRINCIPAL (main) =======

def main():
    """Flujo de ejecución principal del script."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Datos
    df_limpio = cargar_datos_limpios(DATA_PATH)
    
    # 2. Agregar datos por semana
    df_semanal = agregar_datos_por_semana(df_limpio)
    
    if df_semanal.empty:
        print("[ERROR] No se pudieron agregar los datos, el script se detendrá.")
        return
        
    # 3. Generar el gráfico de regresión simple (el que pediste)
    fig_path_1 = FIG_DIR / "01_regresion_lineal_simple.png"
    generar_plot_regresion_simple(df_semanal, fig_path_1)
    
    # 4. Generar el gráfico de estacionalidad (el de tu informe 02)
    fig_path_2 = FIG_DIR / "02_curva_estacional_real.png"
    generar_plot_estacionalidad_real(df_semanal, fig_path_2)
    
    print("\n✅ Proceso de visualización completado.")

# ======= 5. PUNTO DE ENTRADA (Entrypoint) =======

if __name__ == "__main__":
    main()