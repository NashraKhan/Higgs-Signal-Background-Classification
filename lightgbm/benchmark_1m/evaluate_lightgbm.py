
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

X_test = pd.read_csv(DATA_DIR / "X_test.csv", header=None)
y_test = pd.read_csv(DATA_DIR / "y_test.csv", header=None).squeeze("columns")

experiments = {
    "low_level": list(range(0, 21)),
    "derived": list(range(21, 28)),
    "all_features": list(range(0, 28)),
}

results = []

for name, columns in experiments.items():
    model_path = MODEL_DIR / f"lightgbm_{name}.txt"
    model = lgb.Booster(model_file=str(model_path))

    probabilities = model.predict(X_test.iloc[:, columns])
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "experiment": name,
        "features": len(columns),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    })

results_df = pd.DataFrame(results)
RESULTS_DIR.mkdir(exist_ok=True)
results_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)

print("\nFINAL TEST RESULTS")
print(results_df.round(5).to_string(index=False))
print("\nSaved to results/test_results.csv")
