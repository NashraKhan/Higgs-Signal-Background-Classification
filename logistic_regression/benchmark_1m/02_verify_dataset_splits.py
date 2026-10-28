
from pathlib import Path
import sys
import pandas as pd

# ==========================================================
# PHASE 2: HIGGS DATASET SPLIT VERIFICATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# CHANGE THIS if your CSV files are stored elsewhere.

DATA_DIR = Path(
    r"C:\Users\Admin\Documents\Nashra Khan- 2509399 SYBTECH\Nashra - 2509399 SYBTECH - Ensemble\data"
)


CHUNK_SIZE = 100_000

EXPECTED_ROWS = {
    "higgs_train_700k.csv": 700_000,
    "higgs_validation_150k.csv": 150_000,
    "higgs_test_150k.csv": 150_000,
}

EXPECTED_COLUMNS = ["label"] + [
    f"feature_{i}" for i in range(1, 29)
]


def verify_file(filename, expected_rows):
    path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"VERIFYING: {filename}")
    print("=" * 65)
    print(f"Path: {path}")

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Check DATA_DIR and the filename."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    actual_rows = 0
    missing_counts = {}
    invalid_counts = {}
    class_counts = {0: 0, 1: 0}
    invalid_label_count = 0
    actual_columns = None
    chunk_number = 0

    reader = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        chunk_number += 1
        actual_rows += len(chunk)

        print(
            f"Reading chunk {chunk_number}: "
            f"{len(chunk):,} rows"
        )

        if actual_columns is None:
            actual_columns = list(chunk.columns)

            if len(actual_columns) != 29:
                raise ValueError(
                    f"Expected 29 columns; found "
                    f"{len(actual_columns)}."
                )

            if actual_columns != EXPECTED_COLUMNS:
                missing_names = sorted(
                    set(EXPECTED_COLUMNS) - set(actual_columns)
                )
                extra_names = sorted(
                    set(actual_columns) - set(EXPECTED_COLUMNS)
                )

                raise ValueError(
                    "Incorrect column names or order.\n"
                    f"Missing columns: {missing_names}\n"
                    f"Unexpected columns: {extra_names}\n"
                    f"Actual order: {actual_columns}"
                )

            missing_counts = {
                col: 0 for col in actual_columns
            }
            invalid_counts = {
                col: 0 for col in actual_columns
            }

        # Check every column for missing and invalid values.
        for col in actual_columns:
            series = chunk[col]

            missing_counts[col] += int(series.isna().sum())

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            invalid_mask = series.notna() & numeric.isna()

            # Check infinity and negative infinity as well.
            infinite_mask = (
                numeric.notna()
                & ~numeric.map(lambda x: abs(x) != float("inf"))
            )

            invalid_counts[col] += int(
                invalid_mask.sum() + infinite_mask.sum()
            )

        # Validate the target labels.
        labels = pd.to_numeric(
            chunk["label"],
            errors="coerce",
        )

        valid_label_mask = labels.isin([0, 1])

        invalid_label_count += int(
            (~valid_label_mask).sum()
        )

        valid_labels = labels[valid_label_mask].astype(int)
        counts = valid_labels.value_counts()

        for label in (0, 1):
            class_counts[label] += int(counts.get(label, 0))

    # Check expected row count.
    print(f"\nActual rows: {actual_rows:,}")
    print(f"Expected rows: {expected_rows:,}")

    if actual_rows != expected_rows:
        raise ValueError(
            f"Incorrect row count for {filename}."
        )

    print(f"Column count: {len(actual_columns)}")
    print("Column names and order: PASS")

    print("\nClass distribution:")
    for label, name in [(0, "Background"), (1, "Signal")]:
        count = class_counts[label]
        percentage = count / actual_rows * 100

        print(
            f"{name} (label {label}): "
            f"{count:,} ({percentage:.3f}%)"
        )

    total_missing = sum(missing_counts.values())
    total_invalid = sum(invalid_counts.values())

    print("\nData-quality checks:")
    print(f"Missing values: {total_missing:,}")
    print(f"Invalid/non-finite numeric values: {total_invalid:,}")
    print(f"Invalid labels: {invalid_label_count:,}")

    if total_missing:
        print("Columns with missing values:")
        for col, count in missing_counts.items():
            if count:
                print(f"  {col}: {count:,}")

    if total_invalid:
        print("Columns with invalid/non-finite values:")
        for col, count in invalid_counts.items():
            if count:
                print(f"  {col}: {count:,}")

    if total_missing or total_invalid or invalid_label_count:
        raise ValueError(
            f"Data-quality checks failed for {filename}. "
            "The original file has not been modified."
        )

    print(f"\n{filename}: PASS")

    return actual_columns


def main():
    print("=" * 65)
    print("HIGGS LOGISTIC REGRESSION - PHASE 2")
    print("DATASET SPLIT VERIFICATION")
    print("=" * 65)

    print(f"Project directory: {PROJECT_DIR}")
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATA_DIR}\n"
            "Correct DATA_DIR before running again."
        )

    reference_columns = None
    total_rows = 0

    for filename, expected_rows in EXPECTED_ROWS.items():
        columns = verify_file(filename, expected_rows)

        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                f"Column mismatch between files: {filename}"
            )

        total_rows += expected_rows

    print("\n" + "=" * 65)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 65)

    print(f"Training rows:   {EXPECTED_ROWS['higgs_train_700k.csv']:,}")
    print(f"Validation rows: {EXPECTED_ROWS['higgs_validation_150k.csv']:,}")
    print(f"Test rows:       {EXPECTED_ROWS['higgs_test_150k.csv']:,}")
    print(f"Total rows:      {total_rows:,}")
    print("Columns per file: 29")
    print("Features: feature_1 through feature_28")
    print("Target: label")
    print("Column consistency: PASS")

    print("\nPHASE 2 RESULT: ALL CHECKS PASSED")
    print("No dataset was modified. No model was trained.")


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        print("\nPHASE 2 VERIFICATION FAILED")
        print(error)
        print("\nFix the issue before continuing.")
        sys.exit(1)
