
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "results"

experiments = [
    ("Experiment 1 - 21 features",
     "catboost_experiment_1_21_metrics.csv"),
    ("Experiment 2 - 7 features",
     "catboost_experiment_2_7_metrics.csv"),
    ("Experiment 3 - 28 features",
     "catboost_experiment_3_28_metrics.csv"),
]

frames = []

for name, filename in experiments:
    path = RESULT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)
    df.insert(0, "experiment_name", name)
    frames.append(df)

comparison = pd.concat(frames, ignore_index=True)

columns = [
    "experiment_name",
    "number_of_features",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "log_loss",
    "training_time_seconds",
]

comparison = comparison[columns]
comparison.to_csv(
    RESULT_DIR / "catboost_experiment_comparison.csv",
    index=False,
)

print("=" * 75)
print("CATBOOST EXPERIMENT COMPARISON")
print("=" * 75)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

best = comparison.loc[comparison["roc_auc"].idxmax()]
print("\nBest validation ROC-AUC:")
print(f"Experiment: {best['experiment_name']}")
print(f"ROC-AUC: {best['roc_auc']:.6f}")

print(f"\nComparison saved to: {RESULT_DIR / 'catboost_experiment_comparison.csv'}")
print("The test dataset was not accessed.")
