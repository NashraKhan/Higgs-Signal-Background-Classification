from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("HIGGS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
FEATURES = [f"feature_{i}" for i in range(1, 29)]
MODEL_PATHS = {
    "xgboost": ROOT / "xgboost/benchmark_1m/models/experiment_3_xgboost_28_features_5000_rounds.json",
    "catboost": ROOT / "catboost/benchmark_1m/models/catboost_experiment_3_28_features.cbm",
    "logistic_regression": ROOT / "logistic_regression/benchmark_1m/models/higgs_logistic_regression.joblib",
}

app = FastAPI(title="HIGGS Local Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def model_available(name: str) -> bool:
    return name in MODEL_PATHS and MODEL_PATHS[name].is_file()


def model_status() -> list[dict]:
    """Report models that actually load, not merely artifacts that exist."""
    labels = {
        "xgboost": "XGBoost · 5,000 rounds",
        "catboost": "CatBoost",
        "logistic_regression": "Logistic Regression",
    }
    statuses = []
    for name, label in labels.items():
        if not model_available(name):
            missing = str(MODEL_PATHS[name].relative_to(ROOT))
            statuses.append({"id": name, "name": label, "available": False, "ready": False, "error": "Required model artifact missing: " + missing})
            continue
        try:
            load_model(name)
            statuses.append({"id": name, "name": label, "available": True, "ready": True, "error": None})
        except Exception as exc:
            statuses.append({"id": name, "name": label, "available": True, "ready": False, "error": str(exc)})
    return statuses


@lru_cache(maxsize=3)
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
    if name == "logistic_regression":
        import joblib
        return joblib.load(path)
    raise ValueError("Unsupported or disabled model: " + name)


def predict_probability(name: str, values: list[float]) -> float:
    frame = pd.DataFrame([values], columns=FEATURES, dtype=np.float32)
    model = load_model(name)
    probability = float(model.predict_proba(frame)[:, 1][0])
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


def read_source_dataset_sample():
    """Fallback for a genuine model demo when only the original HIGGS.csv is available.

    This is explicitly NOT a held-out test event. Only the first 5,000 rows are read
    so the exhibition demo does not scan the full multi-gigabyte source file.
    """
    candidates = [
        ROOT / "HIGGS.csv", ROOT / "higgs.csv",
        ROOT / "data" / "HIGGS.csv", ROOT / "data" / "higgs.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, header=None, nrows=5000)
            values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            if values.shape[1] != 29 or not np.isfinite(values).all():
                frame = pd.read_csv(path, nrows=5000)
                lower = {str(c).strip().lower(): c for c in frame.columns}
                if all("feature_" + str(i) in lower for i in range(1, 29)) and "label" in lower:
                    x = frame[[lower["feature_" + str(i)] for i in range(1, 29)]].to_numpy(dtype=np.float32)
                    y = frame[lower["label"]].to_numpy(dtype=int)
                else:
                    continue
            else:
                first, last = values[:, 0], values[:, -1]
                if set(np.unique(first)).issubset({0.0, 1.0}):
                    y, x = first.astype(int), values[:, 1:]
                elif set(np.unique(last)).issubset({0.0, 1.0}):
                    x, y = values[:, :28], last.astype(int)
                else:
                    continue
            if len(x) == 0 or x.shape[1] != 28 or not np.isfinite(x).all():
                continue
            index = random.randrange(len(x))
            return index, x[index].astype(float).tolist(), int(y[index]), str(path)
        except Exception:
            continue
    raise FileNotFoundError(
        "No held-out test split or local HIGGS.csv source file was found. "
        "For a held-out event, add X_test.csv/y_test.csv under lightgbm/benchmark_1m/data/. "
        "As a demo fallback, place the original HIGGS.csv in the repository root or data/ folder."
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
        index = random.randrange(len(x))
        return {
            "index": index, "total_events": len(x), "sample_type": "held_out_test",
            "features": x[index].astype(float).tolist(),
            "actual_label": int(y[index]) if y is not None else None, "source": source,
        }
    except FileNotFoundError:
        try:
            index, features, actual_label, source = read_source_dataset_sample()
            return {
                "index": index, "total_events": 5000, "sample_type": "source_dataset_demo",
                "features": features, "actual_label": actual_label, "source": source,
            }
        except FileNotFoundError:
            # Explicitly synthetic fallback: real model inference on generated
            # in-range feature values, NOT a real collision or evaluation event.
            ranges_path = ROOT / "xgboost/benchmark_1m/results/feature_ranges.csv"
            if not ranges_path.is_file():
                raise HTTPException(status_code=404, detail="No dataset or feature_ranges.csv found.")
            ranges = pd.read_csv(ranges_path, index_col=0)
            ranges = ranges.loc[FEATURES]
            low = ranges["Minimum"].to_numpy(dtype=np.float32)
            high = ranges["Maximum"].to_numpy(dtype=np.float32)
            values = np.random.default_rng().uniform(low, high).astype(np.float32)
            return {
                "index": random.randrange(1000000),
                "total_events": 0,
                "sample_type": "synthetic_demo",
                "features": values.astype(float).tolist(),
                "actual_label": None,
                "source": str(ranges_path) + " (feature min/max only; generated input)",
            }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class PredictionRequest(BaseModel):
    model: Literal["xgboost", "catboost", "logistic_regression"]
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
        # Model deserialization / dependency incompatibility is an unavailable
        # inference service for this model, not a successful API connection.
        raise HTTPException(status_code=503, detail="Model is not ready for inference: " + str(exc)) from exc
    label = int(probability >= request.threshold)
    return {
        "model": request.model, "prediction": label,
        "class_name": "Signal" if label == 1 else "Background",
        "p_signal": probability, "p_background": 1.0 - probability,
        "confidence": probability if label else 1.0 - probability,
        "threshold": request.threshold,
    }
