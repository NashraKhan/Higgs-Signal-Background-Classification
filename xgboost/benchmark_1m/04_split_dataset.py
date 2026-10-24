import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# PHASE 4
# SPLIT 1M DEVELOPMENT DATASET INTO:
#
# TRAIN       = 700,000
# VALIDATION  = 150,000
# TEST        = 150,000
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = Path(
    "data/higgs_development_1m.csv"
)

OUTPUT_FOLDER = Path("data")


TRAIN_FILE = OUTPUT_FOLDER / "higgs_train_700k.csv"
VALIDATION_FILE = OUTPUT_FOLDER / "higgs_validation_150k.csv"
TEST_FILE = OUTPUT_FOLDER / "higgs_test_150k.csv"


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42

TRAIN_SIZE = 700_000
VALIDATION_SIZE = 150_000
TEST_SIZE = 150_000

TOTAL_ROWS = 1_000_000


# ------------------------------------------------------------
# 3. Column names
# ------------------------------------------------------------

COLUMN_NAMES = ["label"] + [
    f"feature_{i}" for i in range(1, 29)
]


# ------------------------------------------------------------
# 4. Check input file
# ------------------------------------------------------------

if not INPUT_FILE.exists():

    print("ERROR: Development dataset was not found.")

    print(f"\nExpected file:")
    print(INPUT_FILE)

    print(
        "\nMake sure Phase 3 was completed successfully."
    )

    raise SystemExit


# ------------------------------------------------------------
# 5. Check whether output files already exist
# ------------------------------------------------------------

output_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE
]

existing_files = [
    file for file in output_files
    if file.exists()
]

if existing_files:

    print("ERROR: The following output file(s) already exist:")

    for file in existing_files:
        print(f" - {file}")

    print(
        "\nDelete the existing split files only if you "
        "intentionally want to recreate them."
    )

    raise SystemExit


# ------------------------------------------------------------
# 6. Create output folder
# ------------------------------------------------------------

OUTPUT_FOLDER.mkdir(exist_ok=True)


print("=" * 75)
print("PHASE 4 - TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)


# ------------------------------------------------------------
# 7. Load the 1M development dataset
# ------------------------------------------------------------

print("\nLoading development dataset...")

df = pd.read_csv(
    INPUT_FILE,
    header=None,
    names=COLUMN_NAMES,
    dtype=np.float32
)

print("Development dataset loaded.")

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ------------------------------------------------------------
# 8. Verify size
# ------------------------------------------------------------

if len(df) != TOTAL_ROWS:

    print(
        f"\nERROR: Expected {TOTAL_ROWS:,} rows "
        f"but found {len(df):,}."
    )

    raise SystemExit

print("\nPASS: Development dataset contains exactly 1,000,000 rows.")


# ------------------------------------------------------------
# 9. Separate features and target
# ------------------------------------------------------------

X = df.drop(columns=["label"])

y = df["label"]


# ------------------------------------------------------------
# 10. First split
#
# TRAIN = 700,000
# TEMP  = 300,000
# ------------------------------------------------------------

print("\nCreating training split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    train_size=TRAIN_SIZE,
    stratify=y,
    random_state=RANDOM_SEED,
    shuffle=True
)


# ------------------------------------------------------------
# 11. Second split
#
# VALIDATION = 150,000
# TEST       = 150,000
#
# We split the 300k temporary set into two equal parts.
# ------------------------------------------------------------

print("Creating validation and test splits...")

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=TEST_SIZE,
    train_size=VALIDATION_SIZE,
    stratify=y_temp,
    random_state=RANDOM_SEED,
    shuffle=True
)


# ------------------------------------------------------------
# 12. Reconstruct DataFrames
# ------------------------------------------------------------

train_df = X_train.copy()
train_df.insert(
    0,
    "label",
    y_train
)

validation_df = X_validation.copy()
validation_df.insert(
    0,
    "label",
    y_validation
)

test_df = X_test.copy()
test_df.insert(
    0,
    "label",
    y_test
)


# ------------------------------------------------------------
# 13. Display sizes
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("SPLIT SIZES")
print("=" * 75)

print(
    f"\nTraining set      : {len(train_df):,} rows"
)

print(
    f"Validation set    : {len(validation_df):,} rows"
)

print(
    f"Test set          : {len(test_df):,} rows"
)


# ------------------------------------------------------------
# 14. Check class distributions
# ------------------------------------------------------------

def print_class_distribution(
    name,
    dataset
):
    background = int(
        (dataset["label"] == 0).sum()
    )

    signal = int(
        (dataset["label"] == 1).sum()
    )

    total = len(dataset)

    background_percentage = (
        background / total
    ) * 100

    signal_percentage = (
        signal / total
    ) * 100

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Background (0): "
        f"{background:,} "
        f"({background_percentage:.4f}%)"
    )

    print(
        f"Signal (1):     "
        f"{signal:,} "
        f"({signal_percentage:.4f}%)"
    )

    print(
        f"Total:          "
        f"{total:,}"
    )


print("\n")
print("=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

print_class_distribution(
    "TRAINING SET",
    train_df
)

print_class_distribution(
    "VALIDATION SET",
    validation_df
)

print_class_distribution(
    "TEST SET",
    test_df
)


# ------------------------------------------------------------
# 15. Save the three datasets
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("SAVING DATASETS")
print("=" * 75)


train_df.to_csv(
    TRAIN_FILE,
    index=False,
    header=True
)

print(f"\nCreated: {TRAIN_FILE}")


validation_df.to_csv(
    VALIDATION_FILE,
    index=False,
    header=True
)

print(f"Created: {VALIDATION_FILE}")


test_df.to_csv(
    TEST_FILE,
    index=False,
    header=True
)

print(f"Created: {TEST_FILE}")


# ------------------------------------------------------------
# 16. Final verification
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("FINAL VERIFICATION")
print("=" * 75)


total_split_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)


if len(train_df) == TRAIN_SIZE:
    print("PASS: Training set = 700,000 rows.")
else:
    print("FAIL: Training set size is incorrect.")


if len(validation_df) == VALIDATION_SIZE:
    print("PASS: Validation set = 150,000 rows.")
else:
    print("FAIL: Validation set size is incorrect.")


if len(test_df) == TEST_SIZE:
    print("PASS: Test set = 150,000 rows.")
else:
    print("FAIL: Test set size is incorrect.")


if total_split_rows == TOTAL_ROWS:
    print(
        "PASS: All three splits together = 1,000,000 rows."
    )
else:
    print(
        "FAIL: Split sizes do not add up to 1,000,000."
    )


# ------------------------------------------------------------
# 17. Final message
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 4 COMPLETED")
print("=" * 75)

print("\nCreated files:")

print(f" - {TRAIN_FILE}")
print(f" - {VALIDATION_FILE}")
print(f" - {TEST_FILE}")

print("\nThe original HIGGS.csv was NOT modified.")

print(
    "\nNext phase: PHASE 5 - ML environment and "
    "XGBoost setup."
)

print("\nDo NOT train a model yet.")