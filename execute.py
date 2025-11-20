#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import platform


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def venv_paths(venv_dir: Path):
    """
    Devuelve (python_path, pip_path) dentro del venv según SO.
    """
    if is_windows():
        py = venv_dir / "Scripts" / "python.exe"
        pip = venv_dir / "Scripts" / "pip.exe"
    else:
        py = venv_dir / "bin" / "python"
        pip = venv_dir / "bin" / "pip"
    return py, pip


def ensure_venv_and_deps(root: Path) -> Path:
    """
    1) Crea venv si no existe.
    2) Instala requirements si existen.
    3) Si no está corriendo en venv, relanza este script usando python del venv.
    Retorna la ruta del python del venv.
    """
    venv_dir = root / "venv"  # respetamos tu estructura actual
    venv_py, venv_pip = venv_paths(venv_dir)
    requirements = root / "requirements.txt"

    # 1) Crear venv si falta o está roto
    if not venv_py.exists():
        print(f"🛠️  No existe venv en {venv_dir}. Creándolo...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        venv_py, venv_pip = venv_paths(venv_dir)

    # 2) Instalar dependencias (si requirements existe)
    if requirements.exists():
        print("📦 Instalando/actualizando dependencias desde requirements.txt ...")
        subprocess.run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_py), "-m", "pip", "install", "-r", str(requirements)], check=True)
    else:
        print("⚠️ No existe requirements.txt, se omite instalación de dependencias.")

    # 3) Si NO estamos usando el python del venv, relanzar dentro del venv
    running_in_venv = os.environ.get("RUNNING_IN_VENV") == "1"
    if not running_in_venv:
        current = Path(sys.executable).resolve()
        target = venv_py.resolve()

        if current != target:
            print(f"🔁 Relanzando dentro del venv: {target}")
            env = os.environ.copy()
            env["RUNNING_IN_VENV"] = "1"

            # relanza el mismo execute.py con el python del venv
            result = subprocess.run(
                [str(target), str(root / "execute.py"), *sys.argv[1:]],
                cwd=str(root),
                env=env
            )
            sys.exit(result.returncode)

    return venv_py


def run_script(py_file: Path, python_exec: Path):
    """Ejecuta un script con el python del venv."""
    print(f"\n▶ Ejecutando: {py_file.name}")
    subprocess.run(
        [str(python_exec), str(py_file)],
        cwd=str(py_file.parent),
        check=True
    )


def main():
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    utils_show = root / "utils" / "show_datasets.py"

    if not src_dir.exists():
        raise FileNotFoundError(f"No existe carpeta src en: {src_dir}")
    if not utils_show.exists():
        raise FileNotFoundError(f"No existe show_datasets.py en: {utils_show}")

    # Asegura venv + deps y (si corresponde) relanza dentro del venv
    venv_python = ensure_venv_and_deps(root)

    # 1) Scripts en src ordenados
    scripts = sorted([p for p in src_dir.glob("*.py") if p.name != "__init__.py"])

    if not scripts:
        print("⚠️ No hay scripts .py en src para ejecutar.")
        return

    print("\n🧩 Orden de ejecución en src:")
    for s in scripts:
        print(f"   - {s.name}")

    # 2) Ejecutar secuencialmente
    for script in scripts:
        run_script(script, venv_python)

    # 3) Ejecutar show_datasets al final
    print("\n✅ Ejecutando show_datasets.py (final)...")
    run_script(utils_show, venv_python)

    print("\n🎉 Pipeline completo finalizado OK.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando un script. Código de salida: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ Error general: {e}")
        sys.exit(1)