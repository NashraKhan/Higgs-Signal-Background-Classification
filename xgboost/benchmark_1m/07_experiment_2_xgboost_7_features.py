import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# PHASE 7
# EXPERIMENT 2
#
# XGBoost using the 7 high-level HIGGS features
#
# Features:
# feature_22 -> feature_28
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

TRAIN_FILE = Path(
    "data/higgs_train_700k.csv"
)

VALIDATION_FILE = Path(
    "data/higgs_validation_150k.csv"
)

RESULTS_FOLDER = Path("results")
MODELS_FOLDER = Path("models")

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------

RANDOM_SEED = 42

TARGET_COLUMN = "label"

# Experiment 2 uses ONLY features 22-28
FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(22, 29)
]


# ------------------------------------------------------------
# 3. Check files
# ------------------------------------------------------------

if not TRAIN_FILE.exists():
    print("ERROR: Training file was not found.")
    print(TRAIN_FILE)
    raise SystemExit


if not VALIDATION_FILE.exists():
    print("ERROR: Validation file was not found.")
    print(VALIDATION_FILE)
    raise SystemExit


print("=" * 75)
print("PHASE 7 - EXPERIMENT 2")
print("XGBoost with 7 Features")
print("=" * 75)


# ------------------------------------------------------------
# 4. Experiment configuration
# ------------------------------------------------------------

print("\nExperiment configuration:")
print("-----------------------------------")

print("Algorithm       : XGBoost")
print("Number of       : 7")
print("features")

print("Features used   :")
print(FEATURE_COLUMNS)

print("\nTraining rows   : 700,000")
print("Validation rows : 150,000")

print("\nTest dataset is NOT being used in this phase.")


# ------------------------------------------------------------
# 5. Load training data
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("LOADING TRAINING DATA")
print("=" * 75)

start_load_time = time.perf_counter()

train_df = pd.read_csv(
    TRAIN_FILE
)

training_load_time = (
    time.perf_counter()
    - start_load_time
)

print(
    f"\nTraining data loaded in "
    f"{training_load_time:.2f} seconds."
)

print(
    f"Training shape: {train_df.shape}"
)


# ------------------------------------------------------------
# 6. Load validation data
# ------------------------------------------------------------

print("\nLoading validation data...")

start_load_time = time.perf_counter()

validation_df = pd.read_csv(
    VALIDATION_FILE
)

validation_load_time = (
    time.perf_counter()
    - start_load_time
)

print(
    f"Validation data loaded in "
    f"{validation_load_time:.2f} seconds."
)

print(
    f"Validation shape: {validation_df.shape}"
)


# ------------------------------------------------------------
# 7. Separate features and target
# ------------------------------------------------------------

print("\nPreparing X and y...")

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    TARGET_COLUMN
]

X_validation = validation_df[
    FEATURE_COLUMNS
]

y_validation = validation_df[
    TARGET_COLUMN
]


print("\nTraining feature matrix:")
print(X_train.shape)

print("\nTraining target:")
print(y_train.shape)

print("\nValidation feature matrix:")
print(X_validation.shape)

print("\nValidation target:")
print(y_validation.shape)


# ------------------------------------------------------------
# 8. Remove unnecessary DataFrames
# ------------------------------------------------------------

del train_df
del validation_df


# ------------------------------------------------------------
# 9. Create XGBoost model
#
# Keep the same model configuration as Experiment 1
# for a controlled comparison.
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("CREATING XGBOOST MODEL")
print("=" * 75)

model = XGBClassifier(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    early_stopping_rounds=30,

    n_jobs=-1,

    random_state=RANDOM_SEED
)


print("\nXGBoost model created successfully.")

print("\nModel parameters:")
print("-----------------------------------")

print(f"n_estimators          : {model.n_estimators}")
print(f"max_depth             : {model.max_depth}")
print(f"learning_rate         : {model.learning_rate}")
print(f"subsample             : {model.subsample}")
print(f"colsample_bytree      : {model.colsample_bytree}")
print(f"early_stopping_rounds : {model.early_stopping_rounds}")
print(f"tree_method           : {model.tree_method}")


# ------------------------------------------------------------
# 10. Train model
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("TRAINING MODEL")
print("=" * 75)

print("\nExperiment 2 training has started.")
print("Do not close the terminal while it is running.\n")


training_start = time.perf_counter()

model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_validation, y_validation)
    ],

    verbose=True
)


training_time = (
    time.perf_counter()
    - training_start
)


print("\nTraining completed.")

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)


# ------------------------------------------------------------
# 11. Best iteration
# ------------------------------------------------------------

print("\nBest iteration:")
print(model.best_iteration)

print("\nBest validation score:")
print(model.best_score)


# ------------------------------------------------------------
# 12. Validation predictions
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)


y_validation_probability = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


y_validation_prediction = (
    y_validation_probability >= 0.5
).astype(int)


# ------------------------------------------------------------
# 13. Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_validation,
    y_validation_prediction
)

precision = precision_score(
    y_validation,
    y_validation_prediction,
    zero_division=0
)

recall = recall_score(
    y_validation,
    y_validation_prediction,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    y_validation_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_validation,
    y_validation_probability
)


# ------------------------------------------------------------
# 14. Print metrics
# ------------------------------------------------------------

print("\nValidation metrics:")
print("-----------------------------------")

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1-score  : {f1:.6f}"
)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)


# ------------------------------------------------------------
# 15. Classification report
# ------------------------------------------------------------

print("\nClassification report:")
print("-----------------------------------")

print(
    classification_report(
        y_validation,
        y_validation_prediction,
        digits=6,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 16. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_validation,
    y_validation_prediction
)

print("\nConfusion matrix:")
print("-----------------------------------")

print(cm)


# ------------------------------------------------------------
# 17. ROC curve
# ------------------------------------------------------------

print("\nCreating ROC curve...")

fpr, tpr, thresholds = roc_curve(
    y_validation,
    y_validation_probability
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost - 7 Features (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Experiment 2 - ROC Curve"
)

plt.legend()

plt.tight_layout()


roc_file = (
    RESULTS_FOLDER
    / "experiment_2_roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300
)

plt.close()

print(
    f"ROC curve saved to: {roc_file}"
)


# ------------------------------------------------------------
# 18. Confusion matrix file
# ------------------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

cm_file = (
    RESULTS_FOLDER
    / "experiment_2_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_file
)

print(
    f"Confusion matrix saved to: {cm_file}"
)


# ------------------------------------------------------------
# 19. Save metrics
# ------------------------------------------------------------

metrics = pd.DataFrame({
    "experiment": [
        "Experiment 2"
    ],

    "algorithm": [
        "XGBoost"
    ],

    "features": [
        7
    ],

    "training_rows": [
        len(X_train)
    ],

    "validation_rows": [
        len(X_validation)
    ],

    "accuracy": [
        accuracy
    ],

    "precision": [
        precision
    ],

    "recall": [
        recall
    ],

    "f1_score": [
        f1
    ],

    "roc_auc": [
        roc_auc
    ],

    "training_time_seconds": [
        training_time
    ],

    "best_iteration": [
        model.best_iteration
    ],

    "best_validation_logloss": [
        model.best_score
    ]
})


metrics_file = (
    RESULTS_FOLDER
    / "experiment_2_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)

print(
    f"Metrics saved to: {metrics_file}"
)


# ------------------------------------------------------------
# 20. Save model
# ------------------------------------------------------------

model_file = (
    MODELS_FOLDER
    / "experiment_2_xgboost_7_features.json"
)

model.save_model(
    model_file
)

print(
    f"Model saved to: {model_file}"
)


# ------------------------------------------------------------
# 21. Feature importance
# ------------------------------------------------------------

print("\nCreating feature importance...")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": importance
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)

importance_file = (
    RESULTS_FOLDER
    / "experiment_2_feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print(
    f"Feature importance saved to: "
    f"{importance_file}"
)


# ------------------------------------------------------------
# 22. Print feature importance
# ------------------------------------------------------------

print("\nFeature importance:")
print("-----------------------------------")

print(
    importance_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 23. Final summary
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 7 - EXPERIMENT 2 COMPLETED")
print("=" * 75)

print("\nExperiment:")
print("XGBoost with 7 features")

print("\nValidation results:")
print("-----------------------------------")

print(
    f"Accuracy  : {accuracy:.6f}"
)

print(
    f"Precision : {precision:.6f}"
)

print(
    f"Recall    : {recall:.6f}"
)

print(
    f"F1-score  : {f1:.6f}"
)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)

print(
    f"\nTraining time: "
    f"{training_time / 60:.2f} minutes"
)

print("\nFiles created:")
print(
    f" - {metrics_file}"
)

print(
    f" - {cm_file}"
)

print(
    f" - {roc_file}"
)

print(
    f" - {importance_file}"
)

print(
    f" - {model_file}"
)

print("\nIMPORTANT:")
print("The 150,000-row test dataset was NOT used.")

print("\nNext phase:")
print(
    "PHASE 8 - Experiment 3: "
    "XGBoost with all 28 features"
)

print("\nDo NOT start Phase 8 yet.")