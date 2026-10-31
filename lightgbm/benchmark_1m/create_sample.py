
from pathlib import Path

import duckdb

# Project paths
DATA_DIR = Path("data")
SOURCE_FILE = DATA_DIR / "HIGGS.csv"
OUTPUT_FILE = DATA_DIR / "higgs_sample_1m.csv"

SAMPLE_SIZE = 1_000_000
SEED = 42

if not SOURCE_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE_FILE}")

print("Starting stratified sampling...")
print("Reading the original HIGGS dataset with DuckDB.")

con = duckdb.connect()

try:
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '4GB'")

    # Read the CSV and preserve the original 29-column structure.
    con.execute(
        f"""
        CREATE VIEW higgs AS
        SELECT *
        FROM read_csv(
            '{SOURCE_FILE.resolve().as_posix()}',
            header = false,
            auto_detect = true
        )
        """
    )

    # Sample each class separately to preserve class proportions.
    class_counts = con.execute(
        """
        SELECT column00 AS label, COUNT(*) AS n
        FROM higgs
        GROUP BY column00
        ORDER BY column00
        """
    ).fetchall()

    print("Original class counts:", class_counts)

    # Proportional sample sizes, based on observed class counts.
    total_rows = sum(n for _, n in class_counts)
    sample_counts = {
        label: round(SAMPLE_SIZE * n / total_rows)
        for label, n in class_counts
    }

    # Ensure the requested sample size is exactly one million.
    first_label = class_counts[0][0]
    sample_counts[first_label] += (
        SAMPLE_SIZE - sum(sample_counts.values())
    )

    parts = []
    for label, count in sample_counts.items():
        print(f"Sampling {count:,} rows from class {label}...")
        parts.append(
            con.execute(
                """
                SELECT *
                FROM higgs
                WHERE column00 = ?
                ORDER BY random()
                LIMIT ?
                """,
                [label, count],
            ).df()
        )

    import pandas as pd

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Write without a header so the original format is retained.
    sample.to_csv(OUTPUT_FILE, index=False, header=False)

    print("\nSampling completed!")
    print("Output file:", OUTPUT_FILE)
    print("Sample shape:", sample.shape)
    print("Sample class counts:")
    print(sample.iloc[:, 0].value_counts().sort_index())

finally:
    con.close()
