
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

EXPERIMENT = "catboost_experiment_1_21"
FEATURES = [f"feature_{i}" for i in range(1, 22)]
TARGET = "label"

TRAIN_X = DATA_DIR / "X_train.csv"
TRAIN_Y = DATA_DIR / "y_train.csv"
VAL_X = DATA_DIR / "X_val.csv"
VAL_Y = DATA_DIR / "y_val.csv"

SEED = 42
ITERATIONS = 1500
DEPTH = 8
LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 100

# Leave one CPU core free when possible.
THREADS = max(1, (os.cpu_count() or 2) - 1)


def main():
    print("=" * 65)
    print("EXPERIMENT 1: CATBOOST WITH 21 FEATURES")
    print("=" * 65)

    # --------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------
    for path in [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Data directory: {DATA_DIR}")
    print(f"Features: {FEATURES[0]} through {FEATURES[-1]}")
    print(f"CPU threads: {THREADS}")
    print(f"Maximum iterations: {ITERATIONS}")
    print("Test dataset will not be accessed.")

    # --------------------------------------------------
    # LOAD TRAINING AND VALIDATION DATA
    # Existing CSV files have no header row.
    # --------------------------------------------------
    print("\nLoading training data...")
    X_train = pd.read_csv(TRAIN_X, header=None, usecols=range(21))
    X_train.columns = FEATURES

    y_train_df = pd.read_csv(TRAIN_Y, header=None)
    if y_train_df.shape[1] != 1:
        raise ValueError("Training target must have exactly one column.")
    y_train = y_train_df.iloc[:, 0]

    print("Loading validation data...")
    X_val = pd.read_csv(VAL_X, header=None, usecols=range(21))
    X_val.columns = FEATURES

    y_val_df = pd.read_csv(VAL_Y, header=None)
    if y_val_df.shape[1] != 1:
        raise ValueError("Validation target must have exactly one column.")
    y_val = y_val_df.iloc[:, 0]

    if len(X_train) != 700_000 or len(y_train) != 700_000:
        raise ValueError("Training data does not contain 700,000 rows.")

    if len(X_val) != 150_000 or len(y_val) != 150_000:
        raise ValueError("Validation data does not contain 150,000 rows.")

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be binary (0 and 1).")

    if not set(y_val.unique()).issubset({0, 1}):
        raise ValueError("Validation labels must be binary (0 and 1).")

    if (
        X_train.isnull().values.any()
        or X_val.isnull().values.any()
        or y_train.isnull().any()
        or y_val.isnull().any()
    ):
        raise ValueError("Missing values detected in training/validation data.")

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # --------------------------------------------------
    # TRAIN MODEL
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

    # --------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------
    model_path = MODEL_DIR / f"{EXPERIMENT}_21_features.cbm"
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    # --------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------
    print("\nCalculating validation metrics...")

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "experiment": "Experiment 1 - 21 features",
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

    metrics_path = RESULT_DIR / f"{EXPERIMENT}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("\nValidation metrics:")
    for key in [
        "accuracy", "precision", "recall", "f1_score",
        "roc_auc", "log_loss", "training_time_seconds"
    ]:
        print(f"{key}: {metrics[key]:.6f}")

    # --------------------------------------------------
    # CONFUSION MATRIX
    # Rows = true labels; columns = predicted labels.
    # --------------------------------------------------
    cm = confusion_matrix(y_val, predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["True_0_Background", "True_1_Signal"],
        columns=["Predicted_0_Background", "Predicted_1_Signal"],
    )
    cm_df.to_csv(
        RESULT_DIR / f"{EXPERIMENT}_confusion_matrix.csv"
    )

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------
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

    # --------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------
    fpr, tpr, _ = roc_curve(y_val, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CatBoost Experiment 1 - ROC Curve (Validation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / f"{EXPERIMENT}_roc_curve.png", dpi=150
    )
    plt.close()

    # --------------------------------------------------
    # TRAINING / VALIDATION LOGLOSS CURVE
    # --------------------------------------------------
    history = model.get_evals_result()
    train_loss = history["learn"]["Logloss"]
    val_loss = history["validation"]["Logloss"]

    pd.DataFrame({
        "iteration": range(1, len(train_loss) + 1),
        "training_logloss": train_loss,
        "validation_logloss": val_loss,
    }).to_csv(
        RESULT_DIR / f"{EXPERIMENT}_logloss_history.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss,
             label="Training Logloss")
    plt.plot(range(1, len(val_loss) + 1), val_loss,
             label="Validation Logloss")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Logloss")
    plt.title("CatBoost Experiment 1 - Logloss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / f"{EXPERIMENT}_logloss.png", dpi=150
    )
    plt.close()

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        RESULT_DIR / f"{EXPERIMENT}_feature_importance.csv",
        index=False,
    )

    # Save experiment configuration and feature list.
    metadata = {
        **metrics,
        "features": FEATURES,
        "target": TARGET,
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

    print("\nAll Experiment 1 outputs saved in:")
    print(RESULT_DIR)
    print("\nSUCCESS: Experiment 1 completed.")
    print("The test dataset was not used.")


if __name__ == "__main__":
    main()
