
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
sample_file = DATA_DIR / "higgs_sample_1m.csv"

# The CSV has no header: column 0 is the label, columns 1-28 are features.
df = pd.read_csv(sample_file, header=None)

y = df.iloc[:, 0].astype(int)
X = df.iloc[:, 1:]

# 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Save features and labels separately.
for name, features, labels in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    features.to_csv(DATA_DIR / f"X_{name}.csv", index=False, header=False)
    labels.to_csv(DATA_DIR / f"y_{name}.csv", index=False, header=False)
    print(f"{name}: X={features.shape}, y={labels.shape}")

print("\nSplits saved successfully.")
