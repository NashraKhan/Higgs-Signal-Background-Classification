# HIGGS live inference integration

This folder provides a local Python API for inference using the project's saved HIGGS model artifacts. It does not retrain models or commit private model files or datasets.

## Start the API (from the repository root)

PowerShell:

    python -m pip install -r higgs_inference/requirements.txt
    python -m uvicorn higgs_inference.app:app --host 127.0.0.1 --port 8000

Check model availability: http://127.0.0.1:8000/health

## Event source priority

The API first looks for a genuine held-out test event:

    lightgbm/benchmark_1m/data/X_test.csv
    lightgbm/benchmark_1m/data/y_test.csv

It checks the corresponding xgboost, catboost, neural_network, logistic_regression and root data folders too. X_test.csv must contain 28 feature columns; y_test.csv contains matching 0/1 labels. A combined higgs_test_150k.csv or test.csv with 28 features plus label is also supported.

If no held-out split is present, the API can use the original HIGGS.csv from the repository root or data/ folder. It reads at most the first 5,000 rows and randomly selects one. The frontend explicitly labels this as a source-dataset demo event, not a held-out test event. Do not use this fallback to claim test-set evaluation.

## Saved model availability

The API checks these existing artifacts:
- XGBoost: xgboost/benchmark_1m/models/experiment_3_xgboost_28_features_5000_rounds.json
- CatBoost: catboost/benchmark_1m/models/catboost_experiment_3_28_features.cbm
- LightGBM: lightgbm/benchmark_1m/models/lightgbm_all_features.txt
- Logistic Regression: logistic_regression/benchmark_1m/models/higgs_logistic_regression.joblib
- MLP: neural_network/benchmark_1m/models/mlp_experiment_3.pt plus mlp_experiment_3_scaler.npz

The API reports missing artifacts as unavailable; it never fabricates predictions. The MLP checkpoint is not available if its .pt file is absent locally.

## API
- GET /health: model artifact availability
- GET /sample: a real event row from the held-out test split, or clearly labelled source-dataset fallback
- POST /predict: predict one event with a selected saved model

Prediction JSON body:

    {"model":"xgboost","features":[28 numeric values],"threshold":0.5}

The Grok frontend shell patch is at grok-dashboard-patch/src/components/chamber/shell.tsx. Copy it to the matching path in your separate Higgs-Grok-Dashboard-Review frontend project.
