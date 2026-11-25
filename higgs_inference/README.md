# HIGGS live inference integration

This folder provides a local Python API for the existing saved HIGGS checkpoints.
It does not retrain models and it does not commit datasets or checkpoints.

## Start the API (from the repository root)

PowerShell:

    .venv\Scripts\Activate.ps1
    python -m pip install -r higgs_inference/requirements.txt
    python -m uvicorn higgs_inference.app:app --host 127.0.0.1 --port 8000

Health check: http://127.0.0.1:8000/health

## Test-data requirement

The /sample endpoint uses a real row from the held-out test split. It searches for:

    lightgbm/benchmark_1m/data/X_test.csv
    lightgbm/benchmark_1m/data/y_test.csv

It also checks corresponding xgboost, catboost, neural_network, logistic_regression and root data folders.
Alternatively, it accepts higgs_test_150k.csv (28 features plus label) in one of those data folders.

Dataset files are intentionally excluded from Git. If your test split is stored elsewhere, copy it to
lightgbm/benchmark_1m/data/ as X_test.csv and y_test.csv. X_test must contain 28 feature columns;
y_test must contain the matching binary labels (0=Background, 1=Signal).

## Saved model availability

The API loads the existing 28-feature saved models:
- XGBoost: experiment_3_xgboost_28_features_5000_rounds.json
- CatBoost: catboost_experiment_3_28_features.cbm
- LightGBM: lightgbm_all_features.txt
- Logistic Regression: higgs_logistic_regression.joblib
- MLP: mlp_experiment_3.pt plus mlp_experiment_3_scaler.npz

The MLP .pt checkpoint is ignored by Git in this repository. It must exist in the local
neural_network/benchmark_1m/models/ directory for MLP inference to be enabled. The API reports it
as unavailable otherwise; it will not fabricate an MLP prediction.

## API
- GET /health: model artifact availability
- GET /sample: random held-out test event and label
- POST /predict: predict one event with a selected saved model

Prediction JSON body:
    {"model":"xgboost","features":[28 numeric values],"threshold":0.5}

Run the dashboard frontend in a separate terminal. The updated Grok shell patch uses
http://127.0.0.1:8000 by default.
