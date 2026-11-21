
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# PHASE 5: FINAL TEST DATASET EVALUATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    r"C:\Users\Admin\Documents\Nashra Khan- 2509399 SYBTECH\Nashra - 2509399 SYBTECH - Ensemble\data"
)

TEST_FILE = DATA_DIR / "higgs_test_150k.csv"

MODEL_FILE = PROJECT_DIR / "models" / "higgs_logistic_regression.joblib"

OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_FILE = OUTPUT_DIR / "phase5_test_metrics.json"

FEATURES = [f"feature_{i}" for i in range(1, 29)]
TARGET = "label"


def main():
    print("=" * 65)
    print("HIGGS LOGISTIC REGRESSION - PHASE 5")
    print("FINAL TEST DATASET EVALUATION")
    print("=" * 65)

    # 1. Check required files.
    print("\n1. CHECKING FILES")

    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_FILE}")

    print("Test dataset: FOUND")
    print("Saved model: FOUND")

    # 2. Load the test data.
    print("\n2. LOADING TEST DATA")

    df = pd.read_csv(TEST_FILE)

    expected = set(FEATURES + [TARGET])

    if len(df.columns) != 29 or set(df.columns) != expected:
        raise ValueError(
            "Unexpected test CSV columns.\n"
            f"Actual columns: {df.columns.tolist()}"
        )

    df = df[FEATURES + [TARGET]]

    if df.isna().to_numpy().any():
        raise ValueError("Test dataset contains missing values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Test target must contain only 0 and 1.")

    X_test = df[FEATURES]
    y_test = df[TARGET].astype(int)

    if not np.isfinite(X_test.to_numpy(dtype=np.float64)).all():
        raise ValueError("Test features contain non-finite values.")

    print(f"Test rows: {len(X_test):,}")
    print(f"Feature count: {len(FEATURES)}")
    print("Test data checks: PASS")

    # 3. Load the trained pipeline.
    print("\n3. LOADING TRAINED MODEL")

    model = joblib.load(MODEL_FILE)

    print("Model loading: PASS")

    # 4. Make predictions.
    print("\n4. GENERATING TEST PREDICTIONS")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Predictions: COMPLETE")

    # 5. Calculate final metrics.
    print("\n5. FINAL TEST RESULTS")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(X_test)),
        "feature_count": len(FEATURES),
    }

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
        else:
            print(f"{name.replace('_', ' ').title():12}: {value}")

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Background (0)", "Signal (1)"],
            zero_division=0,
        )
    )

    # 6. Save final metrics.
    print("\n6. SAVING TEST RESULTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved: {METRICS_FILE}")

    # 7. Finish.
    print("\n" + "=" * 65)
    print("PHASE 5 RESULT: TEST EVALUATION COMPLETE")
    print("The saved model was evaluated without retraining.")
    print("=" * 65)


if __name__ == "__main__":
    main()
