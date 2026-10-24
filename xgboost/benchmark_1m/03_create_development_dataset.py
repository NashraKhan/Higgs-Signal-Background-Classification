import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PHASE 3
# CREATE A STRATIFIED 1,000,000-ROW DEVELOPMENT DATASET
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = Path("HIGGS.csv")

OUTPUT_FOLDER = Path("data")
OUTPUT_FILE = OUTPUT_FOLDER / "higgs_development_1m.csv"


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

TOTAL_DEVELOPMENT_ROWS = 1_000_000

RANDOM_SEED = 42

CHUNK_SIZE = 500_000


# ------------------------------------------------------------
# 3. Expected class counts
#
# Based on the full dataset statistics from Phase 2:
#
# Background = 5,170,877
# Signal     = 5,829,123
#
# Therefore for 1,000,000 rows:
#
# Background = 470,080
# Signal     = 529,920
# ------------------------------------------------------------

EXPECTED_BACKGROUND = 470_080
EXPECTED_SIGNAL = 529_920


# ------------------------------------------------------------
# 4. Check input file
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    print("ERROR: HIGGS.csv was not found.")
    print("Make sure HIGGS.csv is in the same folder as this script.")
    raise SystemExit


# ------------------------------------------------------------
# 5. Create output folder
# ------------------------------------------------------------

OUTPUT_FOLDER.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 6. Prevent accidental overwrite
# ------------------------------------------------------------

if OUTPUT_FILE.exists():

    print("ERROR: The development dataset already exists:")
    print(OUTPUT_FILE)

    print("\nDelete the existing file only if you intentionally")
    print("want to recreate the dataset.")

    raise SystemExit


print("=" * 75)
print("PHASE 3 - CREATE STRATIFIED 1M DEVELOPMENT DATASET")
print("=" * 75)


# ------------------------------------------------------------
# 7. Read ONLY the label column
#
# This is much smaller than loading all 29 columns.
# ------------------------------------------------------------

print("\nReading the label column...")

labels = pd.read_csv(
    INPUT_FILE,
    header=None,
    usecols=[0],
    dtype=np.int8
)[0]

print("Label column loaded.")
print(f"Total labels: {len(labels):,}")


# ------------------------------------------------------------
# 8. Find row indices for each class
# ------------------------------------------------------------

background_indices = np.flatnonzero(
    labels.to_numpy() == 0
)

signal_indices = np.flatnonzero(
    labels.to_numpy() == 1
)


print("\nOriginal class counts:")
print("-----------------------------------")
print(f"Background: {len(background_indices):,}")
print(f"Signal    : {len(signal_indices):,}")


# ------------------------------------------------------------
# 9. Stratified random sampling
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


selected_background = rng.choice(
    background_indices,
    size=EXPECTED_BACKGROUND,
    replace=False
)


selected_signal = rng.choice(
    signal_indices,
    size=EXPECTED_SIGNAL,
    replace=False
)


# Combine both classes
selected_indices = np.concatenate(
    [
        selected_background,
        selected_signal
    ]
)


# ------------------------------------------------------------
# 10. Shuffle selected rows
#
# This prevents the output file from being grouped by class.
# ------------------------------------------------------------

rng.shuffle(selected_indices)


# Boolean array telling us which original rows to extract
selected_mask = np.zeros(
    len(labels),
    dtype=bool
)

selected_mask[selected_indices] = True


# We no longer need the full labels Series
del labels

print("\nStratified sampling completed.")

print("\nSelected rows:")
print("-----------------------------------")
print(f"Background: {EXPECTED_BACKGROUND:,}")
print(f"Signal    : {EXPECTED_SIGNAL:,}")
print(f"Total     : {len(selected_indices):,}")


# ------------------------------------------------------------
# 11. Column names
# ------------------------------------------------------------

column_names = ["label"] + [
    f"feature_{i}" for i in range(1, 29)
]


# ------------------------------------------------------------
# 12. Extract selected rows from the original dataset
#
# We process HIGGS.csv in chunks instead of loading all
# 11 million rows into memory at once.
# ------------------------------------------------------------

print("\nExtracting selected rows from HIGGS.csv...")
print("This may take some time.\n")


total_written = 0
output_class_counts = {
    0: 0,
    1: 0
}

first_chunk = True
chunk_number = 0


for chunk_start, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        header=None,
        names=column_names,
        dtype=np.float32,
        chunksize=CHUNK_SIZE
    )
):

    chunk_number += 1

    actual_start = chunk_start * CHUNK_SIZE
    actual_end = actual_start + len(chunk)

    # Which rows from this chunk were selected?
    mask = selected_mask[
        actual_start:actual_end
    ]

    selected_chunk = chunk.loc[mask]

    if len(selected_chunk) > 0:

        # Count classes for verification
        output_class_counts[0] += int(
            (selected_chunk["label"] == 0).sum()
        )

        output_class_counts[1] += int(
            (selected_chunk["label"] == 1).sum()
        )

        total_written += len(selected_chunk)

        # Write to CSV
        selected_chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=False,
            index=False
        )

        first_chunk = False

    print(
        f"Processed chunk {chunk_number} "
        f"| Selected rows written: {total_written:,}"
    )


# ------------------------------------------------------------
# 13. Final verification
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("DEVELOPMENT DATASET VERIFICATION")
print("=" * 75)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nRows written:")
print(f"{total_written:,}")

print("\nClass distribution:")
print("-----------------------------------")
print(
    f"Background (0): "
    f"{output_class_counts[0]:,}"
)

print(
    f"Signal (1):     "
    f"{output_class_counts[1]:,}"
)


# ------------------------------------------------------------
# 14. Verify exact expected counts
# ------------------------------------------------------------

print("\nVerification checks:")
print("-----------------------------------")


if total_written == TOTAL_DEVELOPMENT_ROWS:
    print("PASS: Exactly 1,000,000 rows created.")
else:
    print(
        "FAIL: Incorrect number of rows created."
    )


if output_class_counts[0] == EXPECTED_BACKGROUND:
    print("PASS: Correct number of background events.")
else:
    print("FAIL: Background count is incorrect.")


if output_class_counts[1] == EXPECTED_SIGNAL:
    print("PASS: Correct number of signal events.")
else:
    print("FAIL: Signal count is incorrect.")


if (
    output_class_counts[0]
    + output_class_counts[1]
    == TOTAL_DEVELOPMENT_ROWS
):
    print("PASS: Class counts add up to 1,000,000.")
else:
    print("FAIL: Class counts do not add up correctly.")


# ------------------------------------------------------------
# 15. Final message
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 3 COMPLETED")
print("=" * 75)

print("\nCreated:")
print(OUTPUT_FILE)

print("\nThe original HIGGS.csv was NOT modified.")

print("\nNext phase:")
print("PHASE 4 - Train / Validation / Test split")

print("\nDo NOT start Phase 4 yet.")