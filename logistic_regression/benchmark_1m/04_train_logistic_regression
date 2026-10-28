

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==========================================================
# PHASE 4: HIGGS LOGISTIC REGRESSION TRAINING
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    r"C:\Users\Admin\Documents\Nashra Khan- 2509399 SYBTECH\Nashra - 2509399 SYBTECH - Ensemble\data"
)

TRAIN_FILE = DATA_DIR / "higgs_train_700k.csv"
VALIDATION_FILE = DATA_DIR / "higgs_validation_150k.csv"

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

FEATURES = [f"feature_{i}" for i in range(1, 29)]
TARGET = "label"


def load_dataset(file_path, dataset_name):
    print(f"\nLoading {dataset_name}: {file_path.name}")

    # Read the CSV header and data.
    df = pd.read_csv(file_path)

    print(f"Rows found: {len(df):,}")
    print(f"Columns found: {len(df.columns)}")
    print(f"Actual column names: {df.columns.tolist()}")

    # Check the required columns without assuming their order.
    expected = set(FEATURES + [TARGET])
    actual = set(df.columns)

    missing = expected - actual
    extra = actual - expected

    if missing or extra or len(df.columns) != 29:
        raise ValueError(
            f"\nColumn mismatch in {file_path.name}\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Unexpected columns: {sorted(extra)}\n"
            f"Actual columns: {df.columns.tolist()}"
        )

    # Reorder by column name, regardless of the CSV's original order.
    # This handles both label-first and label-last CSV files.
    df = df[FEATURES + [TARGET]]

    # Validate the target labels.
    if df[TARGET].isna().any():
        raise ValueError(f"{dataset_name} contains missing labels.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{dataset_name} has invalid target labels.")

    # Check feature values.
    feature_df = df[FEATURES]

    if feature_df.isna().to_numpy().any():
        raise ValueError(f"{dataset_name} contains missing features.")

    if not np.isfinite(feature_df.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            f"{dataset_name} contains infinite or invalid feature values."
        )

    X = df[FEATURES]
    y = df[TARGET].astype("int8")

    print(f"{dataset_name} loading: PASS")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    return X, y


def main():
    print("=" * 65)
    print("HIGGS LOGISTIC REGRESSION - PHASE 4")
    print("MODEL TRAINING AND VALIDATION")
    print("=" * 65)

    # 1. Verify paths.
    print("\n1. CHECKING FILES")

    for path in [TRAIN_FILE, VALIDATION_FILE]:
        if not path.is_file():
            raise FileNotFoundError(
                f"File not found:\n{path}\n"
                "Check DATA_DIR before continuing."
            )
        print(f"FOUND: {path.name}")

    # 2. Load training and validation data.
    print("\n2. LOADING DATA")

    X_train, y_train = load_dataset(
        TRAIN_FILE, "Training dataset"
    )

    X_validation, y_validation = load_dataset(
        VALIDATION_FILE, "Validation dataset"
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")

    # 3. Create the pipeline.
    print("\n3. CREATING MODEL")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("Feature scaling: READY")
    print("Logistic Regression: READY")

    # 4. Train.
    print("\n4. TRAINING MODEL")
    print("Training has started. Please wait.")

    start = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start

    print("Training: COMPLETE")
    print(f"Training time: {training_seconds:.2f} seconds")

    # 5. Evaluate on validation data.
    print("\n5. VALIDATION EVALUATION")

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_validation, predictions)
        ),
        "precision": float(
            precision_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_validation, probabilities)
        ),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "feature_count": len(FEATURES),
        "training_seconds": round(training_seconds, 2),
        "scikit_learn_version": sklearn.__version__,
    }

    for name in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]:
        print(f"{name.replace('_', ' ').title():12}: {metrics[name]:.4f}")

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(
        y_validation, predictions, labels=[0, 1]
    ))

    # 6. Save model and validation metrics.
    print("\n6. SAVING RESULTS")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "higgs_logistic_regression.joblib"
    metrics_path = OUTPUT_DIR / "phase4_validation_metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")

    # 7. Completion.
    print("\n" + "=" * 65)
    print("PHASE 4 RESULT: SUCCESS")
    print("Model trained and validation metrics calculated.")
    print("Test dataset has NOT been used.")
    print("=" * 65)


if __name__ == "__main__":
    main()

