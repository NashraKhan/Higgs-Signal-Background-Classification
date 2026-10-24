
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
    roc_curve,
)

# ============================================================
# PHASE 8 - EXPERIMENT 3
# XGBoost with all 28 HIGGS features
# ============================================================

TRAIN_FILE = Path("data/higgs_train_700k.csv")
VALIDATION_FILE = Path("data/higgs_validation_150k.csv")

RESULTS_FOLDER = Path("results")
MODELS_FOLDER = Path("models")

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

RANDOM_SEED = 42
TARGET_COLUMN = "label"

FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 29)
]

# ------------------------------------------------------------
# 1. Check input files
# ------------------------------------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

print("=" * 75)
print("PHASE 8 - EXPERIMENT 3")
print("XGBoost with all 28 features")
print("=" * 75)

# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

print("\nLoading training data...")
load_start = time.perf_counter()

train_df = pd.read_csv(TRAIN_FILE)

print(f"Training shape: {train_df.shape}")
print(f"Training load time: {time.perf_counter() - load_start:.2f}s")

print("\nLoading validation data...")
load_start = time.perf_counter()

validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Validation shape: {validation_df.shape}")
print(
    f"Validation load time: "
    f"{time.perf_counter() - load_start:.2f}s"
)

# ------------------------------------------------------------
# 3. Separate features and target
# ------------------------------------------------------------

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]

print("\nTraining feature matrix:", X_train.shape)
print("Training target:", y_train.shape)
print("Validation feature matrix:", X_validation.shape)
print("Validation target:", y_validation.shape)

del train_df, validation_df

print("\nFeatures used:")
print(FEATURE_COLUMNS)

print("\nThe test dataset is NOT being used.")

# ------------------------------------------------------------
# 4. Configure XGBoost
# Use the same parameters as Experiments 1 and 2
# ------------------------------------------------------------

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
    random_state=RANDOM_SEED,
)

print("\nModel configuration:")
print(model.get_params())

# ------------------------------------------------------------
# 5. Train model
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TRAINING EXPERIMENT 3")
print("=" * 75)

training_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=True,
)

training_time = time.perf_counter() - training_start

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Training time: {training_time / 60:.2f} minutes")
print(f"Best iteration: {model.best_iteration}")
print(f"Best validation logloss: {model.best_score}")

# ------------------------------------------------------------
# 6. Evaluate on validation data
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("VALIDATION EVALUATION")
print("=" * 75)

y_probability = model.predict_proba(X_validation)[:, 1]
y_prediction = (y_probability >= 0.5).astype(int)

accuracy = accuracy_score(y_validation, y_prediction)
precision = precision_score(
    y_validation, y_prediction, zero_division=0
)
recall = recall_score(
    y_validation, y_prediction, zero_division=0
)
f1 = f1_score(
    y_validation, y_prediction, zero_division=0
)
roc_auc = roc_auc_score(y_validation, y_probability)

print(f"\nAccuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")

print("\nClassification report:")
print(
    classification_report(
        y_validation,
        y_prediction,
        digits=6,
        zero_division=0,
    )
)

# ------------------------------------------------------------
# 7. Save confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_validation, y_prediction)

print("\nConfusion matrix:")
print(cm)

cm_file = RESULTS_FOLDER / "experiment_3_confusion_matrix.csv"

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(cm_file)

# ------------------------------------------------------------
# 8. Save ROC curve
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_validation, y_probability)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"XGBoost - 28 Features (AUC = {roc_auc:.4f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Experiment 3 - ROC Curve")
plt.legend()
plt.tight_layout()

roc_file = RESULTS_FOLDER / "experiment_3_roc_curve.png"
plt.savefig(roc_file, dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. Save metrics
# ------------------------------------------------------------

metrics_file = RESULTS_FOLDER / "experiment_3_metrics.csv"

pd.DataFrame([{
    "experiment": "Experiment 3",
    "algorithm": "XGBoost",
    "features": 28,
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "training_time_seconds": training_time,
    "best_iteration": model.best_iteration,
    "best_validation_logloss": model.best_score,
}]).to_csv(metrics_file, index=False)

# ------------------------------------------------------------
# 10. Save feature importance
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_file = RESULTS_FOLDER / "experiment_3_feature_importance.csv"
importance_df.to_csv(importance_file, index=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

# ------------------------------------------------------------
# 11. Save trained model
# ------------------------------------------------------------

model_file = MODELS_FOLDER / "experiment_3_xgboost_28_features.json"
model.save_model(model_file)

# ------------------------------------------------------------
# 12. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PHASE 8 - EXPERIMENT 3 COMPLETED")
print("=" * 75)

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")
print(f"Training time: {training_time / 60:.2f} minutes")

print("\nFiles created:")
print(f" - {metrics_file}")
print(f" - {cm_file}")
print(f" - {roc_file}")
print(f" - {importance_file}")
print(f" - {model_file}")

print("\nThe test dataset was NOT used.")
print("Next: Phase 9 - Compare all three experiments.")
print("Do NOT start Phase 9 yet.")
