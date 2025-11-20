# utils/show_datasets.py
# -*- coding: utf-8 -*-
"""
Lista CSV en outputs/eda, permite elegir uno y exporta una 'tablita' con las
primeras N filas a CSV y HTML (y Markdown si está disponible).

Uso:
  python utils/show_datasets.py
Opciones:
  --n 15           # cuántas filas exportar (por defecto 10)
  --out outputs/eda/_previews   # carpeta de salida (por defecto)
"""

from pathlib import Path
import sys
import argparse
import pandas as pd

EDA_DIR = Path("outputs/eda")

def count_lines_fast(p: Path) -> int:
    with p.open("rb") as f:
        return sum(1 for _ in f)

def list_csvs(base: Path):
    return sorted([p for p in base.glob("*.csv") if p.is_file()])

def read_head(p: Path, n: int) -> pd.DataFrame:
    """Lee solo N filas con autodetección de separador. Prueba encodings comunes."""
    last_err = None
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                p,
                sep=None,              # autodetecta delimitador
                engine="python",
                nrows=n,
                on_bad_lines="skip",
                encoding=enc
            )
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No fue posible leer {p.name}: {last_err}")

def export_table(df: pd.DataFrame, outdir: Path, base_name: str):
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{base_name}.csv"
    html_path = outdir / f"{base_name}.html"
    md_path = outdir / f"{base_name}.md"

    # CSV limpio
    df.to_csv(csv_path, index=False)

    # HTML simple (abre con el navegador)
    df.to_html(html_path, index=False)

    # Markdown (opcional, requiere 'tabulate')
    try:
        df.to_markdown(md_path, index=False)
        md_ok = True
    except Exception:
        md_ok = False

    print("\n✅ Exportado:")
    print(f"  • CSV : {csv_path}")
    print(f"  • HTML: {html_path}")
    if md_ok:
        print(f"  • MD  : {md_path}")
    else:
        print("  • MD  : (omitido; instala 'tabulate' si lo necesitas: pip install tabulate)")

def main():
    parser = argparse.ArgumentParser(description="Exporta una tablita (head N) de un CSV en outputs/eda.")
    parser.add_argument("--n", type=int, default=10, help="Filas a exportar (por defecto 10)")
    parser.add_argument("--out", default="outputs/eda/_previews", help="Carpeta de salida")
    args = parser.parse_args()

    outdir = Path(args.out)

    if not EDA_DIR.exists():
        print(f"[!] No existe {EDA_DIR}.")
        sys.exit(1)

    csvs = list_csvs(EDA_DIR)
    if not csvs:
        print(f"[i] No se encontraron CSV en {EDA_DIR}.")
        sys.exit(0)

    # Tabla simple con conteo de líneas
    print("\n=== CSV disponibles en outputs/eda ===")
    print("{:<3}  {:<30}  {:>14}".format("#", "archivo", "líneas(archivo)"))
    print("{:<3}  {:<30}  {:>14}".format("-", "-"*30, "-"*14))
    for i, p in enumerate(csvs, 1):
        print("{:<3}  {:<30}  {:>14}".format(i, p.name[:30], count_lines_fast(p)))

    while True:
        sel = input("\nElige un archivo por número (Enter para salir): ").strip()
        if sel == "":
            print("Saliendo.")
            return
        if not sel.isdigit() or not (1 <= int(sel) <= len(csvs)):
            print("Número inválido.")
            continue

        p = csvs[int(sel)-1]
        n_str = input(f"¿Cuántas filas exportar de '{p.name}'? [{args.n}]: ").strip()
        n = int(n_str) if (n_str.isdigit() and int(n_str) > 0) else args.n

        try:
            df = read_head(p, n)
        except Exception as e:
            print(f"[!] {e}")
            continue

        base_name = f"{p.stem}_head{n}"
        export_table(df, outdir, base_name)

        # Permite exportar más sin relanzar
        otra = input("\n¿Exportar otro archivo? [s/N]: ").strip().lower()
        if otra != "s":
            print("Listo.")
            return

if __name__ == "__main__":
    main()
