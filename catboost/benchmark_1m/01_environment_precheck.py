
from pathlib import Path
import sys
import shutil
import importlib.metadata

# CatBoost project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Existing datasets stay in the LightGBM project's data folder.
DATA_DIR = PROJECT_DIR.parent / "HIGGS-LightGBM" / "data"

PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "catboost",
    "matplotlib",
]

DATA_FILES = [
    "X_train.csv",
    "y_train.csv",
    "X_val.csv",
    "y_val.csv",
    "X_test.csv",
    "y_test.csv",
    "HIGGS.csv",
    "higgs_sample_1m.csv",
]

def main():
    print("=" * 55)
    print("PHASE 1: CATBOOST ENVIRONMENT PRECHECK")
    print("=" * 55)

    print(f"\nProject folder: {PROJECT_DIR}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    print("\n--- Package availability ---")
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
            print(f"[OK] {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"[MISSING] {package}")

    print("\n--- Dataset files ---")
    print(f"Data folder: {DATA_DIR}")

    if not DATA_DIR.is_dir():
        print("[ERROR] Data folder not found.")
        print("Check the folder names and paths.")
    else:
        for filename in DATA_FILES:
            path = DATA_DIR / filename

            if path.is_file():
                size_mb = path.stat().st_size / (1024 ** 2)
                try:
                    with path.open("rb") as file:
                        file.read(1)
                    print(f"[FOUND] {filename} | {size_mb:.2f} MB")
                except OSError as error:
                    print(f"[ACCESS ERROR] {filename}: {error}")
            else:
                print(f"[NOT FOUND] {filename}")

    print("\n--- Disk space ---")
    try:
        usage = shutil.disk_usage(PROJECT_DIR)
        print(f"Available: {usage.free / (1024 ** 3):.2f} GB")
    except OSError as error:
        print(f"[ERROR] {error}")

    print("\n--- End of precheck ---")
    print("No packages were installed.")
    print("No datasets were modified or copied.")
    print("No model was trained.")
    print("=" * 55)

if __name__ == "__main__":
    main()
