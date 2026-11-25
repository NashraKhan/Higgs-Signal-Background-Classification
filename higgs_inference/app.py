from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("HIGGS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
FEATURES = [f"feature_{i}" for i in range(1, 29)]
MODEL_PATHS = {
    "xgboost": ROOT / "xgboost/benchmark_1m/models/experiment_3_xgboost_28_features_5000_rounds.json",
    "catboost": ROOT / "catboost/benchmark_1m/models/catboost_experiment_3_28_features.cbm",
    "lightgbm": ROOT / "lightgbm/benchmark_1m/models/lightgbm_all_features.txt",
    "logistic_regression": ROOT / "logistic_regression/benchmark_1m/models/higgs_logistic_regression.joblib",
    "mlp": ROOT / "neural_network/benchmark_1m/models/mlp_experiment_3.pt",
}
MLP_SCALER = ROOT / "neural_network/benchmark_1m/models/mlp_experiment_3_scaler.npz"

app = FastAPI(title="HIGGS Local Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HiggsMLP(torch.nn.Module):
    def __init__(self, input_features: int = 28):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_features, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def model_available(name: str) -> bool:
    path = MODEL_PATHS[name]
    return path.is_file() and (name != "mlp" or MLP_SCALER.is_file())


def model_status() -> list[dict]:
    labels = {
        "xgboost": "XGBoost · 5,000 rounds",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "logistic_regression": "Logistic Regression",
        "mlp": "Neural Network (MLP)",
    }
    return [{"id": n, "name": labels[n], "available": model_available(n), "artifact": str(MODEL_PATHS[n].relative_to(ROOT))} for n in MODEL_PATHS]


@lru_cache(maxsize=5)
def load_model(name: str):
    if name not in MODEL_PATHS:
        raise ValueError("Unknown model: " + name)
    if not model_available(name):
        raise FileNotFoundError("Required model artifact is missing: " + str(MODEL_PATHS[name]))

    path = MODEL_PATHS[name]
    if name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    if name == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    if name == "logistic_regression":
        import joblib
        return joblib.load(path)
    if name == "mlp":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = HiggsMLP(28)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with np.load(MLP_SCALER) as z:
            scaler = {"mean": z["mean"].astype(np.float32), "scale": z["scale"].astype(np.float32)}
        return {"model": model, "scaler": scaler}
    raise ValueError("Unsupported model: " + name)


def predict_probability(name: str, values: list[float]) -> float:
    frame = pd.DataFrame([values], columns=FEATURES, dtype=np.float32)
    matrix = frame.to_numpy(dtype=np.float32)
    model = load_model(name)
    if name in {"xgboost", "catboost", "logistic_regression"}:
        probability = float(model.predict_proba(frame)[:, 1][0])
    elif name == "lightgbm":
        probability = float(model.predict(matrix)[0])
    elif name == "mlp":
        mean = model["scaler"]["mean"]
        scale = np.where(model["scaler"]["scale"] == 0, 1.0, model["scaler"]["scale"])
        scaled = (matrix - mean) / scale
        with torch.no_grad():
            probability = float(torch.sigmoid(model["model"](torch.from_numpy(scaled))).cpu().numpy()[0])
    else:
        raise ValueError("Unsupported model: " + name)
    if not np.isfinite(probability):
        raise ValueError("Model returned a non-finite probability.")
    return float(np.clip(probability, 0.0, 1.0))


def read_test_split():
    directories = [
        ROOT / "lightgbm/benchmark_1m/data", ROOT / "xgboost/benchmark_1m/data",
        ROOT / "catboost/benchmark_1m/data", ROOT / "neural_network/benchmark_1m/data",
        ROOT / "logistic_regression/benchmark_1m/data", ROOT / "data",
    ]
    for directory in directories:
        xp, yp = directory / "X_test.csv", directory / "y_test.csv"
        if xp.is_file():
            xf = pd.read_csv(xp, header=None)
            values = xf.to_numpy(dtype=np.float32)
            if values.shape[1] == 29 and not yp.is_file():
                return values[:, :28], values[:, 28].astype(int), str(xp)
            if values.shape[1] != 28:
                raise ValueError(str(xp) + " must contain 28 feature columns.")
            labels = pd.read_csv(yp, header=None).iloc[:, 0].to_numpy(dtype=int) if yp.is_file() else None
            return values, labels, str(xp)
        for combined in (directory / "higgs_test_150k.csv", directory / "test.csv"):
            if combined.is_file():
                df = pd.read_csv(combined)
                lower = {str(c).strip().lower(): c for c in df.columns}
                if all("feature_" + str(i) in lower for i in range(1, 29)) and "label" in lower:
                    x = df[[lower["feature_" + str(i)] for i in range(1, 29)]].to_numpy(dtype=np.float32)
                    y = df[lower["label"]].to_numpy(dtype=int)
                else:
                    raw = pd.read_csv(combined, header=None).to_numpy(dtype=np.float32)
                    if raw.shape[1] != 29:
                        continue
                    x, y = raw[:, :28], raw[:, 28].astype(int)
                return x, y, str(combined)
    raise FileNotFoundError(
        "No local held-out test events found. Place X_test.csv and y_test.csv in "
        "lightgbm/benchmark_1m/data/ (28 feature columns in X_test.csv), or put "
        "higgs_test_150k.csv (28 features plus label) there. Dataset files are not committed to GitHub."
    )


@lru_cache(maxsize=1)
def load_test_split():
    x, y, source = read_test_split()
    if x.ndim != 2 or x.shape[1] != 28 or not np.isfinite(x).all():
        raise ValueError("Test split must contain finite values in 28 feature columns.")
    if y is not None and (len(y) != len(x) or not set(np.unique(y)).issubset({0, 1})):
        raise ValueError("Test labels must match the feature rows and contain only 0/1.")
    return x, y, source


@app.get("/health")
def health():
    return {"status": "ok", "project_root": str(ROOT), "models": model_status()}


@app.get("/sample")
def sample():
    try:
        x, y, source = load_test_split()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    index = random.randrange(len(x))
    return {
        "index": index, "total_events": len(x), "features": x[index].astype(float).tolist(),
        "actual_label": int(y[index]) if y is not None else None, "source": source,
    }


class PredictionRequest(BaseModel):
    model: Literal["xgboost", "catboost", "lightgbm", "logistic_regression", "mlp"]
    features: list[float] = Field(min_length=28, max_length=28)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)


@app.post("/predict")
def predict(request: PredictionRequest):
    values = np.asarray(request.features, dtype=np.float32)
    if values.shape != (28,) or not np.isfinite(values).all():
        raise HTTPException(status_code=422, detail="Provide exactly 28 finite numerical feature values.")
    try:
        probability = predict_probability(request.model, values.tolist())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed: " + str(exc)) from exc
    label = int(probability >= request.threshold)
    return {
        "model": request.model, "prediction": label,
        "class_name": "Signal" if label == 1 else "Background",
        "p_signal": probability, "p_background": 1.0 - probability,
        "confidence": probability if label else 1.0 - probability,
        "threshold": request.threshold,
    }
