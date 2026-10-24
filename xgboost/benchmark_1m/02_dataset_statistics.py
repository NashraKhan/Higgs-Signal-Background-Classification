import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PHASE 2: FULL HIGGS DATASET STATISTICS
# ============================================================

FILE_PATH = Path("HIGGS.csv")

# Number of columns in the HIGGS dataset
EXPECTED_COLUMNS = 29

# We process the dataset in chunks so that the entire
# 11M-row dataset does not need to fit into RAM at once.
CHUNK_SIZE = 500_000


# ------------------------------------------------------------
# 1. Check that the file exists
# ------------------------------------------------------------

if not FILE_PATH.exists():
    print("ERROR: HIGGS.csv was not found.")
    print("Make sure HIGGS.csv is in the same folder as this script.")
    raise SystemExit


print("=" * 75)
print("PHASE 2 - FULL HIGGS DATASET STATISTICS")
print("=" * 75)

print("\nFile:", FILE_PATH)
print("Chunk size:", CHUNK_SIZE)


# ------------------------------------------------------------
# 2. Column names
# ------------------------------------------------------------

column_names = ["label"] + [
    f"feature_{i}" for i in range(1, 29)
]


# ------------------------------------------------------------
# 3. Initialize statistics
# ------------------------------------------------------------

total_rows = 0

signal_count = 0
background_count = 0

missing_values = pd.Series(
    0,
    index=column_names,
    dtype="int64"
)

feature_min = pd.Series(
    np.inf,
    index=column_names,
    dtype="float64"
)

feature_max = pd.Series(
    -np.inf,
    index=column_names,
    dtype="float64"
)


# ------------------------------------------------------------
# 4. Read the complete dataset in chunks
# ------------------------------------------------------------

print("\nStarting full dataset scan...")
print("This may take some time because HIGGS.csv is very large.\n")

chunk_number = 0

for chunk in pd.read_csv(
    FILE_PATH,
    header=None,
    names=column_names,
    chunksize=CHUNK_SIZE,
    dtype=np.float32
):

    chunk_number += 1

    # Number of rows processed
    rows_in_chunk = len(chunk)
    total_rows += rows_in_chunk


    # --------------------------------------------------------
    # Count target classes
    # --------------------------------------------------------

    background_count += int(
        (chunk["label"] == 0).sum()
    )

    signal_count += int(
        (chunk["label"] == 1).sum()
    )


    # --------------------------------------------------------
    # Count missing values
    # --------------------------------------------------------

    missing_values += chunk.isna().sum()


    # --------------------------------------------------------
    # Update minimum values
    # --------------------------------------------------------

    current_min = chunk.min()

    feature_min = pd.concat(
        [feature_min, current_min]
    ).groupby(level=0).min()


    # --------------------------------------------------------
    # Update maximum values
    # --------------------------------------------------------

    current_max = chunk.max()

    feature_max = pd.concat(
        [feature_max, current_max]
    ).groupby(level=0).max()


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print(
        f"Processed chunk {chunk_number} "
        f"| Rows processed: {total_rows:,}"
    )


# ------------------------------------------------------------
# 5. Calculate class percentages
# ------------------------------------------------------------

if total_rows > 0:
    background_percentage = (
        background_count / total_rows
    ) * 100

    signal_percentage = (
        signal_count / total_rows
    ) * 100
else:
    background_percentage = 0
    signal_percentage = 0


# ------------------------------------------------------------
# 6. Print final results
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("FULL DATASET RESULTS")
print("=" * 75)

print(f"\nTotal rows       : {total_rows:,}")
print(f"Total columns    : {EXPECTED_COLUMNS}")

print("\nTarget distribution:")
print("-----------------------------------")

print(
    f"Background (0)  : {background_count:,} "
    f"({background_percentage:.4f}%)"
)

print(
    f"Signal (1)      : {signal_count:,} "
    f"({signal_percentage:.4f}%)"
)


# ------------------------------------------------------------
# 7. Check total class count
# ------------------------------------------------------------

print("\nClass count check:")

if background_count + signal_count == total_rows:
    print("SUCCESS: Every row belongs to class 0 or class 1.")
else:
    other_rows = (
        total_rows
        - background_count
        - signal_count
    )

    print(
        f"WARNING: {other_rows:,} rows have unexpected labels."
    )


# ------------------------------------------------------------
# 8. Missing values
# ------------------------------------------------------------

print("\nMissing values:")
print("-----------------------------------")

total_missing = int(missing_values.sum())

print("Total missing values:", total_missing)

if total_missing == 0:
    print("SUCCESS: No missing values found in the complete dataset.")
else:
    print("\nColumns containing missing values:")
    print(
        missing_values[missing_values > 0]
    )


# ------------------------------------------------------------
# 9. Feature ranges
# ------------------------------------------------------------

print("\nFeature ranges:")
print("-----------------------------------")

range_table = pd.DataFrame({
    "Minimum": feature_min,
    "Maximum": feature_max
})

print(range_table.to_string())


# ------------------------------------------------------------
# 10. Save statistics
# ------------------------------------------------------------

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

range_table.to_csv(
    output_folder / "feature_ranges.csv"
)

summary = pd.DataFrame({
    "Metric": [
        "Total Rows",
        "Total Columns",
        "Background Count",
        "Signal Count",
        "Background Percentage",
        "Signal Percentage",
        "Total Missing Values"
    ],
    "Value": [
        total_rows,
        EXPECTED_COLUMNS,
        background_count,
        signal_count,
        background_percentage,
        signal_percentage,
        total_missing
    ]
})

summary.to_csv(
    output_folder / "dataset_summary.csv",
    index=False
)


# ------------------------------------------------------------
# 11. Final message
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 2 COMPLETED")
print("=" * 75)

print("\nFiles created:")
print("results/dataset_summary.csv")
print("results/feature_ranges.csv")

print("\nDo NOT start model training yet.")
print("Wait for the Phase 2 results before moving to Phase 3.")