
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "HIGGS-LightGBM" / "data"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_experiment_3_28_features.cbm"
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [f"feature_{i}" for i in range(1, 29)]

TEST_X = DATA_DIR / "X_test.csv"
TEST_Y = DATA_DIR / "y_test.csv"


def main():
    print("=" * 65)
    print("FINAL TEST EVALUATION: CATBOOST WITH 28 FEATURES")
    print("=" * 65)

    for path in [MODEL_PATH, TEST_X, TEST_Y]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load the selected model.
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"Loaded model: {MODEL_PATH}")

    # Load test data only now, after model selection.
    print("\nLoading test dataset for final evaluation...")
    X_test = pd.read_csv(TEST_X, header=None)
    X_test.columns = FEATURES
    y_df = pd.read_csv(TEST_Y, header=None)

    if y_df.shape[1] != 1:
        raise ValueError("Test target must contain exactly one column.")

    y_test = y_df.iloc[:, 0]

    if X_test.shape != (150_000, 28) or len(y_test) != 150_000:
        raise ValueError(
            f"Unexpected test shapes: X={X_test.shape}, y={y_test.shape}"
        )

    if (
        X_test.isnull().values.any()
        or y_test.isnull().any()
        or not np.isfinite(X_test.to_numpy()).all()
    ):
        raise ValueError("Missing or infinite values found in test data.")

    if set(y_test.unique()) != {0, 1}:
        raise ValueError("Test labels must be binary: 0 and 1.")

    print(f"Test shape: {X_test.shape}")

    # Evaluate the chosen model once on the test set.
    start = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    prediction_seconds = time.perf_counter() - start

    metrics = {
        "selected_model": MODEL_PATH.name,
        "number_of_features": len(FEATURES),
        "test_rows": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
        "prediction_time_seconds": prediction_seconds,
        "threshold": 0.5,
    }

    output_csv = RESULT_DIR / "catboost_final_test_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["True_0_Background", "True_1_Signal"],
        columns=["Predicted_0_Background", "Predicted_1_Signal"],
    ).to_csv(RESULT_DIR / "catboost_final_test_confusion_matrix.csv")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Background_0", "Signal_1"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / "catboost_final_test_classification_report.csv"
    )

    with open(
        RESULT_DIR / "catboost_final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nFINAL TEST METRICS")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            print(f"{key}: {value:.6f}" if isinstance(value, (float, np.floating))
                  else f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print(f"\nMetrics saved to: {output_csv}")
    print("SUCCESS: Final test evaluation completed.")


if __name__ == "__main__":
    main()
