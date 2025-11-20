# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

OUT = Path("outputs/report"); OUT.mkdir(parents=True, exist_ok=True)

def main():
    met = Path("outputs/forecast/metricas_forecast.csv")
    metrics = pd.read_csv(met) if met.exists() else pd.DataFrame()

    txt = []
    txt.append("# Interpretabilidad y Consideraciones Éticas\n")
    txt.append("## Cómo leer el modelo\n")
    txt.append("- El modelo SARIMAX incorpora **tendencia** y **estacionalidad semanal (~52)**.")
    txt.append("- Los intervalos de confianza (IC95%) reflejan la **incertidumbre** esperada.\n")
    txt.append("## Por qué sube o baja la serie\n")
    txt.append("- **Invierno** (jun–ago): aumento estacional por circulación de virus respiratorios.\n"
               "- **Verano**: descenso relativo.\n"
               "- **Cambios estructurales**: pandemia, campañas de vacunación, y variabilidad climática (frentes fríos) pueden alterar patrones.\n")
    txt.append("## Factores externos relevantes (no incluidos todavía)\n")
    txt.append("- **Clima**: temperatura mínima, amplitud térmica, humedad.\n- **Epidemiología**: circulación de influenza/RSV/COVID.\n"
               "- **Demografía/movilidad**: población por comuna, flujos temporales.\n")
    txt.append("## Métricas de desempeño (holdout)\n")
    if not metrics.empty:
        txt.append(metrics.to_markdown(index=False))
    else:
        txt.append("- (No encontré `metricas_forecast.csv`)\n")
    txt.append("\n## Riesgos y limitaciones\n"
               "- **Datos agregados**: no contienen PII, pero pueden existir subregistros.\n"
               "- **Sesgo temporal**: cambios de codificación o coberturas entre años.\n"
               "- **Uso responsable**: el pronóstico **no es diagnóstico clínico**; sirve para **planificación**.\n")
    txt.append("## Recomendaciones operativas\n"
               "- Reforzar dotación en semanas pico invernales.\n"
               "- Monitorear desvíos >20% respecto al pronóstico para activar alertas.\n"
               "- Incorporar variables climáticas como **exógenas** en la siguiente iteración.\n")

    (OUT / "interpretabilidad_etica.md").write_text("\n".join(txt), encoding="utf-8")
    print("✅ interpretabilidad_etica.md generado en outputs/report/")

if __name__ == "__main__":
    main()
