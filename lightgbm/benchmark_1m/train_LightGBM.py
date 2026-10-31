
from pathlib import Path
import json
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load the existing train and validation splits.
X_train = pd.read_csv(DATA_DIR / "X_train.csv", header=None)
y_train = pd.read_csv(DATA_DIR / "y_train.csv", header=None).squeeze("columns")

X_val = pd.read_csv(DATA_DIR / "X_val.csv", header=None)
y_val = pd.read_csv(DATA_DIR / "y_val.csv", header=None).squeeze("columns")

experiments = {
    "low_level": list(range(0, 21)),   # Features 1–21
    "derived": list(range(21, 28)),    # Features 22–28
    "all_features": list(range(0, 28)),
}

results = []

for name, columns in experiments.items():
    print(f"\n{'=' * 55}")
    print(f"Training: {name}")
    print(f"Number of features: {len(columns)}")

    start = time.time()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train.iloc[:, columns],
        y_train,
        eval_set=[(X_val.iloc[:, columns], y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    val_prob = model.predict_proba(
        X_val.iloc[:, columns],
        num_iteration=model.best_iteration_,
    )[:, 1]

    val_pred = (val_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, val_prob)
    accuracy = accuracy_score(y_val, val_pred)
    elapsed = time.time() - start

    model_path = MODEL_DIR / f"lightgbm_{name}.txt"
    model.booster_.save_model(str(model_path))

    result = {
        "experiment": name,
        "features": len(columns),
        "best_iteration": model.best_iteration_,
        "validation_auc": float(auc),
        "validation_accuracy": float(accuracy),
        "training_seconds": round(elapsed, 2),
        "model_path": str(model_path),
    }
    results.append(result)

    print(f"Best iteration: {model.best_iteration_}")
    print(f"Validation AUC: {auc:.5f}")
    print(f"Validation accuracy: {accuracy:.5f}")
    print(f"Training time: {elapsed:.1f} seconds")
    print(f"Saved model: {model_path}")

with open(RESULTS_DIR / "validation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nAll training experiments completed.")
print(pd.DataFrame(results).to_string(index=False))
