
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

DATA = Path("data")
MODELS = Path("models")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

X_test = pd.read_csv(DATA / "X_test.csv", header=None)
y_test = pd.read_csv(DATA / "y_test.csv", header=None).squeeze("columns")
test_results = pd.read_csv(RESULTS / "test_results.csv")

experiments = {
    "low_level": list(range(21)),
    "derived": list(range(21, 28)),
    "all_features": list(range(28)),
}

# 1. Compare the three experiments.
metrics = ["accuracy", "f1_score", "roc_auc"]
test_results.set_index("experiment")[metrics].plot(
    kind="bar", figsize=(9, 5), ylim=(0, 1)
)
plt.title("LightGBM Test Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS / "model_comparison.png", dpi=200)
plt.close()

# 2. ROC curves for all three models.
plt.figure(figsize=(8, 6))

for name, columns in experiments.items():
    model = lgb.Booster(model_file=str(MODELS / f"lightgbm_{name}.txt"))
    probabilities = model.predict(X_test.iloc[:, columns])
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LightGBM ROC Curves — Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "roc_curves.png", dpi=200)
plt.close()

# 3. Confusion matrix for the best model.
best_model = lgb.Booster(
    model_file=str(MODELS / "lightgbm_all_features.txt")
)
best_probabilities = best_model.predict(X_test)
best_predictions = (best_probabilities >= 0.5).astype(int)

cm = confusion_matrix(y_test, best_predictions)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Background (0)", "Signal (1)"],
).plot(cmap="Blues", values_format="d")

plt.title("All Features — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
plt.close()

# 4. Feature importance for the best model.
importance = pd.DataFrame({
    "feature": [f"Feature {i}" for i in range(1, 29)],
    "importance": best_model.feature_importance(importance_type="gain"),
})
importance = importance.sort_values("importance", ascending=False)
importance.to_csv(RESULTS / "feature_importance.csv", index=False)

top = importance.head(15).sort_values("importance")
plt.figure(figsize=(9, 7))
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance (gain)")
plt.title("Top 15 Features — All-Features LightGBM")
plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=200)
plt.close()

print("All visualizations and feature importance saved:")
for filename in [
    "model_comparison.png",
    "roc_curves.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "feature_importance.csv",
]:
    print(RESULTS / filename)
