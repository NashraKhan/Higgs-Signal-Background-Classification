
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
)

# ============================================================
# PHASE 13
# EXPERIMENT 1 - EXTENDED TRAINING UP TO 5,000 ROUNDS
# Features: feature_1 to feature_21
# ============================================================

TRAIN_FILE = Path("data/higgs_train_700k.csv")
VALIDATION_FILE = Path("data/higgs_validation_150k.csv")

RESULTS_FOLDER = Path("results")
MODELS_FOLDER = Path("models")

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 22)]
TARGET_COLUMN = "label"

MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100
RANDOM_SEED = 42


# ------------------------------------------------------------
# 1. Verify files
# ------------------------------------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALIDATION_FILE}"
    )

print("=" * 75)
print("PHASE 13 - EXPERIMENT 1")
print("21 features | Up to 5,000 boosting rounds")
print("=" * 75)


# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

print("\nLoading training data...")
train_df = pd.read_csv(TRAIN_FILE, dtype=np.float32)
print("Training shape:", train_df.shape)

print("\nLoading validation data...")
validation_df = pd.read_csv(VALIDATION_FILE, dtype=np.float32)
print("Validation shape:", validation_df.shape)


# ------------------------------------------------------------
# 3. Prepare features and target
# ------------------------------------------------------------

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN].astype(np.int8)

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN].astype(np.int8)

del train_df, validation_df

print("\nTraining feature matrix:", X_train.shape)
print("Validation feature matrix:", X_validation.shape)
print("Number of features:", len(FEATURE_COLUMNS))


# ------------------------------------------------------------
# 4. Configure the model
# ------------------------------------------------------------

model = XGBClassifier(
    n_estimators=MAX_ROUNDS,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    n_jobs=-1,
    random_state=RANDOM_SEED,
)

print("\nMaximum rounds:", MAX_ROUNDS)
print("Early stopping patience:", EARLY_STOPPING_ROUNDS)


# ------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------

print("\nStarting extended training...")
start_time = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=200,
)

training_time = time.perf_counter() - start_time

history = model.evals_result()["validation_0"]["logloss"]
best_round = model.best_iteration + 1
best_logloss = float(model.best_score)

print("\nTraining completed.")
print(f"Training time: {training_time:.2f} seconds")
print(f"Best boosting round: {best_round}")
print(f"Best validation log-loss: {best_logloss:.6f}")


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

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"ROC-AUC   : {roc_auc:.6f}")


# ------------------------------------------------------------
# 7. Save metrics
# ------------------------------------------------------------

metrics_file = RESULTS_FOLDER / "experiment_1_5000_rounds_metrics.csv"

metrics = pd.DataFrame([{
    "experiment": "Experiment 1 - 5000-round limit",
    "algorithm": "XGBoost",
    "features": 21,
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "training_time_seconds": training_time,
    "best_round": best_round,
    "best_validation_logloss": best_logloss,
}])

metrics.to_csv(metrics_file, index=False)


# ------------------------------------------------------------
# 8. Save confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_validation, y_prediction)

cm_file = (
    RESULTS_FOLDER / "experiment_1_5000_rounds_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
).to_csv(cm_file)


# ------------------------------------------------------------
# 9. Plot validation log-loss
# ------------------------------------------------------------

loss_file = RESULTS_FOLDER / "experiment_1_5000_rounds_logloss.png"

plt.figure(figsize=(9, 6))
plt.plot(range(1, len(history) + 1), history)

plt.axvline(
    best_round,
    linestyle="--",
    label=f"Best round: {best_round}",
)

plt.xlabel("Boosting round")
plt.ylabel("Validation log-loss")
plt.title("Experiment 1 - Extended Training to 5,000 Rounds")
plt.legend()
plt.tight_layout()
plt.savefig(loss_file, dpi=300)
plt.close()


# ------------------------------------------------------------
# 10. Save feature importance
# ------------------------------------------------------------

importance_file = (
    RESULTS_FOLDER / "experiment_1_5000_rounds_feature_importance.csv"
)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df.to_csv(importance_file, index=False)


# ------------------------------------------------------------
# 11. Save model separately
# ------------------------------------------------------------

model_file = (
    MODELS_FOLDER / "experiment_1_xgboost_21_features_5000_rounds.json"
)

model.save_model(model_file)


# ------------------------------------------------------------
# 12. Final status
# ------------------------------------------------------------

print("\nFiles created:")
print(f" - {metrics_file}")
print(f" - {cm_file}")
print(f" - {loss_file}")
print(f" - {importance_file}")
print(f" - {model_file}")

if best_round == MAX_ROUNDS:
    print("\nWARNING: The run reached 5,000 rounds.")
    print("Inspect the loss curve before deciding on further training.")
else:
    print("\nEarly stopping selected a round before 5,000.")

print("\nPHASE 13 COMPLETED")
print("The test dataset was NOT used.")
print("Do not start the next phase yet.")
