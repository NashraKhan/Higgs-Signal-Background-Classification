
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    roc_curve,
)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "HIGGS-LightGBM" / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "catboost_experiment_2_7"
FEATURES = [f"feature_{i}" for i in range(22, 29)]

TRAIN_X = DATA_DIR / "X_train.csv"
TRAIN_Y = DATA_DIR / "y_train.csv"
VAL_X = DATA_DIR / "X_val.csv"
VAL_Y = DATA_DIR / "y_val.csv"

SEED = 42
ITERATIONS = 1500
DEPTH = 8
LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 100
THREADS = max(1, (os.cpu_count() or 2) - 1)


def main():
    print("=" * 65)
    print("EXPERIMENT 2: CATBOOST WITH 7 FEATURES")
    print("=" * 65)
    print(f"Data directory: {DATA_DIR}")
    print(f"Features: {FEATURES[0]} through {FEATURES[-1]}")
    print(f"CPU threads: {THREADS}")
    print(f"Maximum iterations: {ITERATIONS}")
    print("Test dataset will not be accessed.")

    # Validate file availability.
    for path in [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load only the required feature columns from headerless CSV files.
    print("\nLoading training data...")
    X_train = pd.read_csv(
        TRAIN_X, header=None, usecols=range(21, 28)
    )
    X_train.columns = FEATURES
    y_train_df = pd.read_csv(TRAIN_Y, header=None)

    print("Loading validation data...")
    X_val = pd.read_csv(
        VAL_X, header=None, usecols=range(21, 28)
    )
    X_val.columns = FEATURES
    y_val_df = pd.read_csv(VAL_Y, header=None)

    if y_train_df.shape[1] != 1 or y_val_df.shape[1] != 1:
        raise ValueError("Target files must contain exactly one column.")

    y_train = y_train_df.iloc[:, 0]
    y_val = y_val_df.iloc[:, 0]

    if X_train.shape != (700_000, 7):
        raise ValueError(f"Unexpected training shape: {X_train.shape}")
    if X_val.shape != (150_000, 7):
        raise ValueError(f"Unexpected validation shape: {X_val.shape}")
    if len(y_train) != 700_000 or len(y_val) != 150_000:
        raise ValueError("Unexpected target row counts.")

    if (
        X_train.isnull().values.any()
        or X_val.isnull().values.any()
        or y_train.isnull().any()
        or y_val.isnull().any()
    ):
        raise ValueError("Missing values detected.")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training features contain infinite values.")
    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("Validation features contain infinite values.")

    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # --------------------------------------------------
    # TRAIN CATBOOST
    # --------------------------------------------------
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=SEED,
        thread_count=THREADS,
        verbose=100,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        allow_writing_files=False,
    )

    print("\nStarting CatBoost training...")
    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=100,
    )

    training_seconds = time.perf_counter() - start_time
    best_iteration = model.get_best_iteration()
    best_val_logloss = model.get_best_score()["validation"]["Logloss"]

    print("\nTraining completed.")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Best iteration (zero-based): {best_iteration}")
    print(f"Best validation Logloss: {best_val_logloss:.6f}")

    # Save model.
    model_path = MODEL_DIR / f"{EXPERIMENT}_7_features.cbm"
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    # --------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------
    print("\nCalculating validation metrics...")
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "experiment": "Experiment 2 - 7 features",
        "number_of_features": len(FEATURES),
        "validation_rows": len(y_val),
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "log_loss": log_loss(y_val, probabilities, labels=[0, 1]),
        "training_time_seconds": training_seconds,
        "best_iteration_zero_based": best_iteration,
        "best_validation_logloss": best_val_logloss,
        "random_seed": SEED,
        "iterations_configured": ITERATIONS,
        "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "thread_count": THREADS,
    }

    pd.DataFrame([metrics]).to_csv(
        RESULT_DIR / f"{EXPERIMENT}_metrics.csv", index=False
    )

    print("\nValidation metrics:")
    for key in [
        "accuracy", "precision", "recall", "f1_score",
        "roc_auc", "log_loss", "training_time_seconds"
    ]:
        print(f"{key}: {metrics[key]:.6f}")

    # Confusion matrix: rows are true classes; columns are predictions.
    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["True_0_Background", "True_1_Signal"],
        columns=["Predicted_0_Background", "Predicted_1_Signal"],
    ).to_csv(RESULT_DIR / f"{EXPERIMENT}_confusion_matrix.csv")

    # Classification report for both classes.
    report = classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Background_0", "Signal_1"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / f"{EXPERIMENT}_classification_report.csv"
    )

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_val, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CatBoost Experiment 2 - ROC Curve (Validation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / f"{EXPERIMENT}_roc_curve.png", dpi=150
    )
    plt.close()

    # Training and validation Logloss history.
    history = model.get_evals_result()
    train_loss = history["learn"]["Logloss"]
    val_loss = history["validation"]["Logloss"]

    pd.DataFrame({
        "iteration": range(1, len(train_loss) + 1),
        "training_logloss": train_loss,
        "validation_logloss": val_loss,
    }).to_csv(
        RESULT_DIR / f"{EXPERIMENT}_logloss_history.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss,
             label="Training Logloss")
    plt.plot(range(1, len(val_loss) + 1), val_loss,
             label="Validation Logloss")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Logloss")
    plt.title("CatBoost Experiment 2 - Logloss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / f"{EXPERIMENT}_logloss.png", dpi=150)
    plt.close()

    # Feature importance.
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        RESULT_DIR / f"{EXPERIMENT}_feature_importance.csv", index=False
    )

    # Configuration and metadata.
    metadata = {
        **metrics,
        "features": FEATURES,
        "target": "label",
        "model_path": str(model_path),
        "training_file": TRAIN_X.name,
        "training_target_file": TRAIN_Y.name,
        "validation_file": VAL_X.name,
        "validation_target_file": VAL_Y.name,
        "test_dataset_used": False,
    }

    with open(
        RESULT_DIR / f"{EXPERIMENT}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nAll Experiment 2 outputs saved in:")
    print(RESULT_DIR)
    print("\nSUCCESS: Experiment 2 completed.")
    print("The test dataset was not used.")


if __name__ == "__main__":
    main()
