
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "HIGGS-LightGBM"
    / "data"
)

SPLITS = {
    "TRAIN": (700_000, "X_train.csv", "y_train.csv"),
    "VALIDATION": (150_000, "X_val.csv", "y_val.csv"),
    "TEST": (150_000, "X_test.csv", "y_test.csv"),
}

FEATURES = [f"feature_{i}" for i in range(1, 29)]


def verify_split(name, expected_rows, x_file, y_file):
    x_path = DATA_DIR / x_file
    y_path = DATA_DIR / y_file

    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"{name}: missing file: {x_path} or {y_path}"
        )

    print(f"\n--- {name} ---")

    # These CSV files have no header row.
    X = pd.read_csv(x_path, header=None)
    y_df = pd.read_csv(y_path, header=None)

    # Verify dimensions before assigning column names.
    if X.shape != (expected_rows, 28):
        raise ValueError(
            f"{x_file}: expected ({expected_rows}, 28), "
            f"found {X.shape}"
        )

    if y_df.shape != (expected_rows, 1):
        raise ValueError(
            f"{y_file}: expected ({expected_rows}, 1), "
            f"found {y_df.shape}"
        )

    X.columns = FEATURES
    y = y_df.iloc[:, 0]

    # Verify numeric values.
    if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError(f"{x_file} contains non-numeric features")

    if not pd.api.types.is_numeric_dtype(y.dtype):
        raise ValueError(f"{y_file} contains non-numeric labels")

    # Check missing and infinite values.
    if X.isnull().values.any() or y.isnull().any():
        raise ValueError(f"{name} contains missing values")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"{x_file} contains infinite values")

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError(f"{y_file} contains infinite values")

    # Verify binary labels.
    labels = set(y.unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected labels 0 and 1; found {labels}")

    print(f"Rows: {len(X):,}")
    print(f"Feature columns: {X.shape[1]}")
    print("Feature names: feature_1 through feature_28")
    print(f"Target file: {y_file}")
    print("Class distribution (%):")
    print(
        (y.value_counts(normalize=True).sort_index() * 100)
        .round(3)
        .to_string()
    )
    print("[PASS] Dimensions, numeric data, labels, missing values")

    del X, y_df, y


def main():
    print("=" * 55)
    print("HIGGS DATASET SPLIT VERIFICATION")
    print("=" * 55)
    print(f"Data directory: {DATA_DIR}")

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    for name, (rows, x_file, y_file) in SPLITS.items():
        verify_split(name, rows, x_file, y_file)

    print("\nSUCCESS: All three splits passed verification.")
    print("No files were modified.")


if __name__ == "__main__":
    main()
