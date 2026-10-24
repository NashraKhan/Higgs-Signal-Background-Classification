import sys
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import xgboost
import matplotlib
import wandb


# ============================================================
# PHASE 5
# ML ENVIRONMENT CHECK
# ============================================================


# ------------------------------------------------------------
# 1. Print Python version
# ------------------------------------------------------------

print("=" * 75)
print("PHASE 5 - ML ENVIRONMENT CHECK")
print("=" * 75)

print("\nPython version:")
print(sys.version)


# ------------------------------------------------------------
# 2. Print library versions
# ------------------------------------------------------------

print("\nInstalled library versions:")
print("-----------------------------------")

print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"XGBoost     : {xgboost.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"W&B         : {wandb.__version__}")


# ------------------------------------------------------------
# 3. Check important project files
# ------------------------------------------------------------

print("\nChecking project files:")
print("-----------------------------------")


required_files = [
    Path("data/higgs_train_700k.csv"),
    Path("data/higgs_validation_150k.csv"),
    Path("data/higgs_test_150k.csv"),
]


all_files_exist = True


for file_path in required_files:

    if file_path.exists():

        size_mb = file_path.stat().st_size / (
            1024 * 1024
        )

        print(
            f"PASS: {file_path} "
            f"({size_mb:.2f} MB)"
        )

    else:

        print(
            f"FAIL: {file_path} "
            f"was not found."
        )

        all_files_exist = False


# ------------------------------------------------------------
# 4. Test XGBoost import
# ------------------------------------------------------------

print("\nTesting XGBoost:")
print("-----------------------------------")

try:

    model = xgboost.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    print("PASS: XGBClassifier created successfully.")

except Exception as error:

    print("FAIL: XGBClassifier could not be created.")

    print("\nError:")
    print(error)

    raise SystemExit


# ------------------------------------------------------------
# 5. Test a tiny model
#
# IMPORTANT:
# This is NOT one of your project experiments.
#
# We only use 100 artificial rows to make sure that
# XGBoost itself can train successfully on this computer.
# ------------------------------------------------------------

print("\nRunning tiny XGBoost test...")
print("-----------------------------------")

try:

    X_test = np.random.default_rng(42).normal(
        size=(100, 5)
    )

    y_test = np.random.default_rng(42).integers(
        0,
        2,
        size=100
    )

    tiny_model = xgboost.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    tiny_model.fit(
        X_test,
        y_test
    )

    predictions = tiny_model.predict(
        X_test[:5]
    )

    print("PASS: XGBoost successfully trained.")
    print("Test predictions:", predictions.tolist())

except Exception as error:

    print("FAIL: XGBoost test training failed.")

    print("\nError:")
    print(error)

    raise SystemExit


# ------------------------------------------------------------
# 6. Final status
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("PHASE 5 RESULT")
print("=" * 75)


if all_files_exist:

    print("\nPASS: Required train/validation/test files exist.")

else:

    print(
        "\nWARNING: One or more required dataset files "
        "are missing."
    )


print("PASS: Python ML environment is working.")
print("PASS: XGBoost is installed.")
print("PASS: XGBoost training test succeeded.")

print("\nNo HIGGS model was trained in this phase.")

print("\n")
print("=" * 75)
print("PHASE 5 COMPLETED")
print("=" * 75)

print("\nNext phase:")
print("PHASE 6 - Experiment 1: XGBoost with 21 features")

print("\nDo NOT start Experiment 1 yet.")