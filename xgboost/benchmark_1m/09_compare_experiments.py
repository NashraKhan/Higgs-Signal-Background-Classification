
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 9 - COMPARE ALL THREE XGBOOST EXPERIMENTS
#
# Experiment 1: 21 features
# Experiment 2: 7 features
# Experiment 3: 28 features
#
# This script does NOT train models.
# This script does NOT use the test dataset.
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

RESULTS_FOLDER = Path("results")

INPUT_FILES = [
    RESULTS_FOLDER / "experiment_1_metrics.csv",
    RESULTS_FOLDER / "experiment_2_metrics.csv",
    RESULTS_FOLDER / "experiment_3_metrics.csv",
]

OUTPUT_TABLE = RESULTS_FOLDER / "experiment_comparison.csv"
METRICS_GRAPH = RESULTS_FOLDER / "experiment_comparison_metrics.png"
TIME_GRAPH = RESULTS_FOLDER / "experiment_training_time_comparison.png"


# ------------------------------------------------------------
# 2. Verify that all results exist
# ------------------------------------------------------------

print("=" * 75)
print("PHASE 9 - EXPERIMENT COMPARISON")
print("=" * 75)

missing_files = [
    file_path for file_path in INPUT_FILES
    if not file_path.exists()
]

if missing_files:
    print("\nERROR: The following results files are missing:")

    for file_path in missing_files:
        print(f" - {file_path}")

    print("\nComplete Experiments 1, 2, and 3 first.")
    raise SystemExit(1)

print("\nPASS: All three experiment results files exist.")


# ------------------------------------------------------------
# 3. Load and combine metrics
# ------------------------------------------------------------

experiment_names = [
    "Experiment 1 (21 features)",
    "Experiment 2 (7 features)",
    "Experiment 3 (28 features)",
]

frames = []

for name, file_path in zip(experiment_names, INPUT_FILES):
    df = pd.read_csv(file_path)

    if len(df) != 1:
        print(f"\nERROR: Expected exactly one metrics row in {file_path}")
        raise SystemExit(1)

    required_columns = [
        "features",
        "training_rows",
        "validation_rows",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
        "best_iteration",
        "best_validation_logloss",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"\nERROR: Missing columns in {file_path}:")
        print(missing_columns)
        raise SystemExit(1)

    df["experiment_name"] = name
    frames.append(df)

comparison = pd.concat(frames, ignore_index=True)

comparison = comparison[
    [
        "experiment_name",
        "features",
        "training_rows",
        "validation_rows",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
        "best_iteration",
        "best_validation_logloss",
    ]
]


# ------------------------------------------------------------
# 4. Save comparison table
# ------------------------------------------------------------

comparison.to_csv(OUTPUT_TABLE, index=False)

print(f"\nComparison table saved: {OUTPUT_TABLE}")


# ------------------------------------------------------------
# 5. Display readable comparison table
#
# Metrics are displayed as percentages here.
# The saved CSV retains the original decimal values.
# ------------------------------------------------------------

display_table = comparison[
    [
        "experiment_name",
        "features",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
    ]
].copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]:
    display_table[column] = display_table[column].map(
        lambda value: f"{value * 100:.2f}%"
    )

display_table["training_time_seconds"] = (
    display_table["training_time_seconds"].map(
        lambda value: f"{value:.2f} seconds"
    )
)

print("\n")
print("=" * 75)
print("COMPARISON TABLE")
print("=" * 75)

print(display_table.to_string(index=False))


# ------------------------------------------------------------
# 6. Compare predictive metrics
# ------------------------------------------------------------

metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]

plot_data = comparison.set_index("experiment_name")[
    metric_columns
] * 100

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 7),
)

ax.set_title("XGBoost Experiment Comparison - Validation Metrics")
ax.set_xlabel("Experiment")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.25)
ax.legend(
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ],
    loc="lower right",
)

plt.tight_layout()
plt.savefig(METRICS_GRAPH, dpi=300)
plt.close()

print(f"\nMetrics graph saved: {METRICS_GRAPH}")


# ------------------------------------------------------------
# 7. Compare training time
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    comparison["experiment_name"],
    comparison["training_time_seconds"],
)

ax.set_title("XGBoost Experiment Comparison - Training Time")
ax.set_xlabel("Experiment")
ax.set_ylabel("Training time (seconds)")
ax.tick_params(axis="x", rotation=10)
ax.grid(axis="y", alpha=0.25)

for bar in bars:
    height = bar.get_height()

    ax.annotate(
        f"{height:.2f}s",
        (bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
    )

plt.tight_layout()
plt.savefig(TIME_GRAPH, dpi=300)
plt.close()

print(f"Training time graph saved: {TIME_GRAPH}")


# ------------------------------------------------------------
# 8. Identify the strongest validation ROC-AUC
# ------------------------------------------------------------

best_index = comparison["roc_auc"].idxmax()
best_row = comparison.loc[best_index]

print("\n")
print("=" * 75)
print("BEST VALIDATION ROC-AUC")
print("=" * 75)

print(f"Experiment : {best_row['experiment_name']}")
print(f"Features   : {int(best_row['features'])}")
print(f"ROC-AUC    : {best_row['roc_auc']:.6f}")
print(f"Accuracy   : {best_row['accuracy']:.6f}")
print(f"F1-score   : {best_row['f1_score']:.6f}")


# ------------------------------------------------------------
# 9. Check whether any experiment reached its iteration cap
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("TRAINING ITERATION CHECK")
print("=" * 75)

print(
    comparison[
        [
            "experiment_name",
            "best_iteration",
            "best_validation_logloss",
        ]
    ].to_string(index=False)
)

if (comparison["best_iteration"] == 499).all():
    print(
        "\nNOTE: All three models reached iteration 499 "
        "out of the configured 500."
    )
    print(
        "Early stopping did not reduce the boosting-round "
        "count in these runs."
    )
    print(
        "We should consider the iteration limit before "
        "finalizing the test evaluation."
    )


# ------------------------------------------------------------
# 10. Final status
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 9 COMPLETED")
print("=" * 75)

print("\nCreated files:")
print(f" - {OUTPUT_TABLE}")
print(f" - {METRICS_GRAPH}")
print(f" - {TIME_GRAPH}")

print("\nNo models were trained in this phase.")
print("The test dataset was NOT accessed.")

print("\nNext: review the comparison and training-round limit.")
