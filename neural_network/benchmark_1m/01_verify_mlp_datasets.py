
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

FILES = {
    "Training": ("higgs_train_700k.csv", 700_000),
    "Validation": ("higgs_validation_150k.csv", 150_000),
    "Test": ("higgs_test_150k.csv", 150_000),
}

EXPECTED_COLUMNS = [
    "label",
    *[f"feature_{i}" for i in range(1, 29)],
]

print("=" * 60)
print("PYTORCH MLP - DATASET VERIFICATION")
print("=" * 60)

for name, (filename, expected_rows) in FILES.items():
    path = DATA_DIR / filename

    print(f"\nChecking {name}: {filename}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    columns = pd.read_csv(path, nrows=0).columns.tolist()

    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {filename}: "
            f"found {len(columns)} columns."
        )

    row_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=["label"],
        chunksize=100_000,
    ):
        if not chunk["label"].isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {filename}")

        row_count += len(chunk)

    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(columns)}")

    if row_count != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"but found {row_count:,}"
        )

    print("PASS")

print("\nALL THREE DATASETS VERIFIED SUCCESSFULLY")
