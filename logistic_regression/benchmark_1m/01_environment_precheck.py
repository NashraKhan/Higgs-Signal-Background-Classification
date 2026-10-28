
import os
import sys
import shutil
import importlib
from pathlib import Path

# ==========================================================
# PHASE 1: ENVIRONMENT PRECHECK
# HIGGS BOSON SIGNAL VS BACKGROUND CLASSIFICATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Update this path if your CSV files are stored elsewhere.
DATA_DIR = PROJECT_DIR / "data"

DATA_FILES = [
    "higgs_train_700k.csv",
    "higgs_validation_150k.csv",
    "higgs_test_150k.csv",
]

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
}


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def check_python():
    print_section("1. PYTHON ENVIRONMENT")

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python path    : {sys.executable}")
    print(f"Project folder : {PROJECT_DIR}")

    virtual_env = os.environ.get("VIRTUAL_ENV")

    if virtual_env:
        print(f"Active venv    : {virtual_env}")
    else:
        print("Active venv    : No VIRTUAL_ENV variable detected")

    if sys.version_info < (3, 10):
        print("WARNING: Python is older than 3.10.")
        print("We will check package compatibility before proceeding.")
    else:
        print("Python version check: Suitable starting point.")


def check_packages():
    print_section("2. REQUIRED PYTHON PACKAGES")

    missing = []

    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "Version unavailable")
            print(f"[OK] {package_name}: {version}")
        except ImportError:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)
        except Exception as error:
            print(f"[ERROR] {package_name}: {error}")
            missing.append(package_name)

    if missing:
        print("\nMissing or unavailable packages:")
        for package in missing:
            print(f"  - {package}")

        print("\nInstall missing packages inside the active venv using:")
        print(
            'python -m pip install '
            + " ".join(missing)
        )
    else:
        print("\nAll required packages are importable.")

    return not missing


def check_datasets():
    print_section("3. DATASET FILE CHECK")

    print(f"Configured data folder: {DATA_DIR}")

    if not DATA_DIR.exists():
        print("[WARNING] The configured data folder does not exist.")
        print("This may be because your CSV files are stored elsewhere.")
        print("No folder or dataset has been created or modified.")
        return False

    all_accessible = True

    for filename in DATA_FILES:
        file_path = DATA_DIR / filename

        print(f"\nFile: {filename}")
        print(f"Path: {file_path}")

        if not file_path.exists():
            print("[MISSING] File not found.")
            all_accessible = False
            continue

        if not file_path.is_file():
            print("[ERROR] The path is not a regular file.")
            all_accessible = False
            continue

        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 ** 2)
        size_gb = size_bytes / (1024 ** 3)

        print(f"Size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

        if size_gb >= 1:
            print(f"      ({size_gb:.2f} GB)")

        try:
            with file_path.open("rb") as file:
                file.read(1)

            print("[OK] File exists and can be opened for reading.")

        except OSError as error:
            print(f"[ERROR] Cannot read file: {error}")
            all_accessible = False

    return all_accessible


def check_disk_space():
    print_section("4. AVAILABLE DISK SPACE")

    try:
        disk_usage = shutil.disk_usage(PROJECT_DIR)

        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)

        print(f"Total disk space: {total_gb:.2f} GB")
        print(f"Free disk space : {free_gb:.2f} GB")

        if free_gb < 5:
            print("WARNING: Less than 5 GB is free.")
            print("Check disk capacity before running large experiments.")
        else:
            print("Disk space check completed.")

    except OSError as error:
        print(f"[ERROR] Could not inspect disk space: {error}")


def main():
    print("=" * 65)
    print("HIGGS LOGISTIC REGRESSION - PHASE 1 PRECHECK")
    print("=" * 65)

    check_python()
    packages_ok = check_packages()
    datasets_ok = check_datasets()
    check_disk_space()

    print_section("5. PRECHECK SUMMARY")

    print(f"Python packages available: {'YES' if packages_ok else 'NO'}")
    print(f"Configured dataset files accessible: {'YES' if datasets_ok else 'NO'}")

    if packages_ok and datasets_ok:
        print("\nEnvironment and configured dataset paths passed basic checks.")
        print("Proceed to Phase 2 only after reviewing the output.")
    else:
        print("\nPrecheck is incomplete. Review the warnings above.")
        print("Resolve missing packages or confirm the dataset folder path.")
        print("Do not proceed to model training.")

    print("\nThis script did not train a model or modify any dataset.")


if __name__ == "__main__":
    main()
