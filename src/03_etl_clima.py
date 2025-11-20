import pandas as pd

df = pd.read_csv(
    "data/2014-2025_Temperaturas_Antofagasta_Calama.csv",
    sep=None,
    engine="python"
)

df = df.rename(columns={
    "Instante": "datetime",
    "Ts": "temperatura",
    "IDCiudad": "ciudad"
})

df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime", "temperatura"])

df["anio"] = df["datetime"].dt.isocalendar().year
df["semanaestadistica"] = df["datetime"].dt.isocalendar().week

# Promedio semanal por ciudad
df_sem = (
    df.groupby(["ciudad", "anio", "semanaestadistica"])["temperatura"]
      .mean()
      .reset_index()
)

# Guardar archivo con Antofagasta y Calama
df_sem.to_csv("data/clima_anio_semana.csv", index=False)

print(df_sem.head())
print("✔ Archivo generado: data/clima_anio_semana.csv")
