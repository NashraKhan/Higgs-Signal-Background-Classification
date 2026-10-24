import pandas as pd
from pathlib import Path


# ============================================================
# PHASE 1: HIGGS DATASET INSPECTION
# ============================================================

# ------------------------------------------------------------
# 1. File location
# ------------------------------------------------------------

FILE_PATH = Path("HIGGS.csv")

if not FILE_PATH.exists():
    print("ERROR: HIGGS.csv was not found.")
    print("Make sure HIGGS.csv is in the same folder as this Python file.")
    raise SystemExit


print("=" * 70)
print("HIGGS BOSON PROJECT - DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# 2. Read a small sample
# ------------------------------------------------------------

print("\nReading first 5 rows...")

df = pd.read_csv(
    FILE_PATH,
    header=None,
    nrows=5
)


# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

print("\nDataset sample shape:")
print(df.shape)

print("\nNumber of columns found:")
print(df.shape[1])


# ------------------------------------------------------------
# 4. Display the first 5 rows
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.to_string(index=False, header=False))


# ------------------------------------------------------------
# 5. Check number of columns
# ------------------------------------------------------------

EXPECTED_COLUMNS = 29

print("\nChecking column count...")

if df.shape[1] == EXPECTED_COLUMNS:
    print("SUCCESS: 29 columns detected.")
else:
    print(
        f"WARNING: Expected {EXPECTED_COLUMNS} columns, "
        f"but found {df.shape[1]} columns."
    )


# ------------------------------------------------------------
# 6. Assign temporary column names
# ------------------------------------------------------------

column_names = ["label"] + [
    f"feature_{i}" for i in range(1, 29)
]

df.columns = column_names


# ------------------------------------------------------------
# 7. Display column names
# ------------------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 8. Inspect the target column
# ------------------------------------------------------------

print("\nTarget column preview:")

print(
    df["label"].value_counts()
)


# ------------------------------------------------------------
# 9. Inspect data types
# ------------------------------------------------------------

print("\nData types:")

print(df.dtypes)


# ------------------------------------------------------------
# 10. Check for missing values in the sample
# ------------------------------------------------------------

print("\nMissing values in the 5-row sample:")

print(df.isnull().sum().sum())


# ------------------------------------------------------------
# 11. Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)

print(f"File found       : {FILE_PATH}")
print(f"Columns detected : {df.shape[1]}")
print(f"Expected columns : {EXPECTED_COLUMNS}")
print("Target column    : label")
print("Features         : feature_1 to feature_28")

print("\nPhase 1 inspection completed.")