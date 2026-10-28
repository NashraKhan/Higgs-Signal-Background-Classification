
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================================
# PHASE 6: LOGISTIC REGRESSION VISUALIZATIONS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
METRICS_FILE = OUTPUT_DIR / "phase5_test_metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not METRICS_FILE.is_file():
    raise FileNotFoundError(
        f"Test metrics not found: {METRICS_FILE}"
    )

with METRICS_FILE.open("r", encoding="utf-8") as file:
    metrics = json.load(file)

print("=" * 60)
print("HIGGS LOGISTIC REGRESSION - PHASE 6")
print("GENERATING PROJECT GRAPHS")
print("=" * 60)

# 1. Performance metrics bar chart.
names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

values = [metrics[key] for key in keys]

plt.figure(figsize=(9, 5))
bars = plt.bar(names, values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("HIGGS Logistic Regression - Test Performance")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "phase6_test_performance.png",
    dpi=300,
)
plt.close()

print("Saved: phase6_test_performance.png")

# 2. Confusion matrix using your actual Phase 5 results.
# Rows: actual class; columns: predicted class.
cm = np.array([
    [37222, 33290],
    [20323, 59165],
])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Background (0)", "Signal (1)"],
)

display.plot(values_format=",d")
plt.title("HIGGS Logistic Regression - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "phase6_confusion_matrix.png",
    dpi=300,
)
plt.close()

print("Saved: phase6_confusion_matrix.png")

# 3. Class-wise precision, recall, and F1-score.
class_names = ["Background (0)", "Signal (1)"]

class_metrics = {
    "Precision": [0.65, 0.64],
    "Recall": [0.53, 0.74],
    "F1-score": [0.58, 0.69],
}

x = np.arange(len(class_names))
width = 0.24

plt.figure(figsize=(9, 5))

for i, (metric_name, scores) in enumerate(class_metrics.items()):
    plt.bar(
        x + (i - 1) * width,
        scores,
        width,
        label=metric_name,
    )

plt.xticks(x, class_names)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Class-wise Test Performance")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "phase6_class_metrics.png",
    dpi=300,
)
plt.close()

print("Saved: phase6_class_metrics.png")

print("\n" + "=" * 60)
print("PHASE 6 RESULT: VISUALIZATIONS COMPLETE")
print(f"Graphs saved in: {OUTPUT_DIR}")
print("=" * 60)
