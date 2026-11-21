from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "lightgbm" / "benchmark_1m" / "data"
OUT_DIR = Path(__file__).resolve().parent
FEATURES = [f"feature_{i}" for i in range(1, 29)]

EXPECTED_TEST_AUC = {
    "XGBoost (5000 rounds)": 0.8372,
    "PyTorch MLP": 0.8360,
    "CatBoost": 0.8321,
}
AUC_CHECK_TOLERANCE = 0.0002
BATCH_SIZE = 8192

PHYSICS_NAMES = {
    "feature_1": "lepton_pT", "feature_2": "lepton_eta", "feature_3": "lepton_phi",
    "feature_4": "missing_energy_mag", "feature_5": "missing_energy_phi",
    "feature_6": "jet_1_pt", "feature_7": "jet_1_eta", "feature_8": "jet_1_phi", "feature_9": "jet_1_b_tag",
    "feature_10": "jet_2_pt", "feature_11": "jet_2_eta", "feature_12": "jet_2_phi", "feature_13": "jet_2_b_tag",
    "feature_14": "jet_3_pt", "feature_15": "jet_3_eta", "feature_16": "jet_3_phi", "feature_17": "jet_3_b_tag",
    "feature_18": "jet_4_pt", "feature_19": "jet_4_eta", "feature_20": "jet_4_phi", "feature_21": "jet_4_b_tag",
    "feature_22": "m_jj (High-Level)", "feature_23": "m_jjj (High-Level)", "feature_24": "m_lv (High-Level)",
    "feature_25": "m_jlv (High-Level)", "feature_26": "m_bb (High-Level)",
    "feature_27": "m_wbb (High-Level)", "feature_28": "m_wwbb (High-Level)",
}
HIGH_LEVEL_SET = {f"feature_{i}" for i in range(22, 29)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Calibri"],
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
})


class HiggsMLP(nn.Module):
    """Must match neural_network/benchmark_1m/04_retrain_mlp_all_experiments.py."""
    def __init__(self, input_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def load_test_split():
    x_path = DATA_DIR / "X_test.csv"
    y_path = DATA_DIR / "y_test.csv"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            "Expected the existing held-out split files: "
            f"{x_path} and {y_path}"
        )

    X = pd.read_csv(x_path, header=None, names=FEATURES, dtype=np.float32)
    y = pd.read_csv(y_path, header=None, dtype=np.int32).iloc[:, 0].to_numpy()

    if X.shape != (150_000, 28):
        raise ValueError(f"Expected X_test shape (150000, 28), found {X.shape}")
    if y.shape != (150_000,):
        raise ValueError(f"Expected y_test shape (150000,), found {y.shape}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, found {np.unique(y)}")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("X_test contains NaN or infinite values")
    return X, y


def load_mlp_probabilities(X: pd.DataFrame) -> np.ndarray:
    model_dir = REPO_ROOT / "neural_network" / "benchmark_1m" / "models"
    checkpoint_path = model_dir / "mlp_experiment_3.pt"
    scaler_path = model_dir / "mlp_experiment_3_scaler.npz"

    if not checkpoint_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            f"Required 28-feature MLP checkpoint/scaler not found: "
            f"{checkpoint_path}, {scaler_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("MLP checkpoint does not contain model_state_dict")
    if checkpoint.get("input_features") != 28:
        raise ValueError(f"Expected a 28-feature MLP checkpoint, got {checkpoint.get('input_features')}")

    feature_numbers = checkpoint.get("feature_numbers")
    if feature_numbers is not None and list(feature_numbers) != list(range(1, 29)):
        raise ValueError(f"Unexpected MLP feature list: {feature_numbers}")

    with np.load(scaler_path) as scaler_data:
        mean = scaler_data["mean"].astype(np.float32)
        scale = scaler_data["scale"].astype(np.float32)

    if mean.shape != (28,) or scale.shape != (28,):
        raise ValueError(f"Expected 28-value scaler arrays, got {mean.shape} and {scale.shape}")
    if np.any(scale == 0):
        raise ValueError("MLP scaler contains a zero scale value")

    X_scaled = ((X.to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)

    model = HiggsMLP(input_features=28)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    batches = []
    with torch.inference_mode():
        for start in range(0, len(X_scaled), BATCH_SIZE):
            xb = torch.from_numpy(X_scaled[start:start + BATCH_SIZE])
            logits = model(xb)
            batches.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def load_all_model_probabilities(X: pd.DataFrame):
    results_dir = REPO_ROOT / "xgboost" / "benchmark_1m" / "results"
    xgb_path = REPO_ROOT / "xgboost" / "benchmark_1m" / "models" / "experiment_3_xgboost_28_features_5000_rounds.json"
    cb_path = REPO_ROOT / "catboost" / "benchmark_1m" / "models" / "catboost_experiment_3_28_features.cbm"

    if not xgb_path.is_file():
        raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
    if not cb_path.is_file():
        raise FileNotFoundError(f"CatBoost model not found: {cb_path}")

    # XGBoost: use the saved 28-feature, 5000-round model.
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))
    try:
        dtest = xgb.DMatrix(X, feature_names=FEATURES)
        xgb_probs = booster.predict(dtest)
    except ValueError:
        # Some saved boosters do not retain feature names; the column order remains 1–28.
        dtest = xgb.DMatrix(X.to_numpy(dtype=np.float32))
        xgb_probs = booster.predict(dtest)

    # PyTorch: load the exact 28-feature checkpoint and its saved scaler.
    mlp_probs = load_mlp_probabilities(X)

    # CatBoost: DataFrame retains feature names and their original order.
    cb = CatBoostClassifier()
    cb.load_model(str(cb_path))
    cb_probs = cb.predict_proba(X)[:, 1]

    probabilities = {
        "XGBoost (5000 rounds)": np.asarray(xgb_probs, dtype=np.float64),
        "PyTorch MLP": np.asarray(mlp_probs, dtype=np.float64),
        "CatBoost": np.asarray(cb_probs, dtype=np.float64),
    }

    print("\n========== CHECK AGAINST VERIFIED HELD-OUT AUC VALUES ==========")
    for name, probs in probabilities.items():
        if probs.shape != (len(X),) or not np.isfinite(probs).all():
            raise ValueError(f"Invalid probability output for {name}: shape={probs.shape}")
        auc = roc_auc_score(Y_TEST, probs)
        expected = EXPECTED_TEST_AUC[name]
        difference = abs(auc - expected)
        print(f"{name}: recomputed AUC={auc:.6f} | report={expected:.4f} | difference={difference:.6f}")
        if difference > AUC_CHECK_TOLERANCE:
            raise RuntimeError(
                f"{name} AUC does not match the verified report within "
                f"{AUC_CHECK_TOLERANCE}. No charts were generated. Investigate the model/data version."
            )
    return probabilities


def plot_overlaid_roc(y_test: np.ndarray, probabilities: dict):
    styles = {
        "XGBoost (5000 rounds)": "#1D4ED8",
        "PyTorch MLP": "#059669",
        "CatBoost": "#D97706",
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
    axins = ax.inset_axes([0.46, 0.14, 0.48, 0.44])

    for label, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        full_label = f"{label} (AUC = {auc_val:.4f})"
        ax.plot(fpr, tpr, label=full_label, color=styles[label], linewidth=2.2)
        axins.plot(fpr, tpr, color=styles[label], linewidth=2.0)

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.2,
            label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate (Background Efficiency)", fontsize=11, fontweight="bold", color="#1E293B")
    ax.set_ylabel("True Positive Rate (Signal Efficiency)", fontsize=11, fontweight="bold", color="#1E293B")
    ax.set_title("Overlaid ROC Curves — All 28 Features (150k Held-Out Test Set)", fontsize=13, fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.0)

    axins.set_xlim(0.10, 0.45)
    axins.set_ylim(0.55, 0.90)
    axins.set_title("Zoomed Operating Region", fontsize=8.5, fontweight="bold", color="#334155")
    axins.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    axins.tick_params(labelsize=8)

    fig.tight_layout()
    out_file = OUT_DIR / "01_overlaid_roc_curves_28_features.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def plot_ablation_progression():
    experiments = [
        "Exp 1: 21 Low-Level\n(Kinematic)",
        "Exp 2: 7 High-Level\n(Physics Mass)",
        "Exp 3: All 28\n(Combined)",
    ]
    x = np.arange(len(experiments))
    width = 0.25
    auc_data = {
        "PyTorch MLP": [0.7676, 0.7911, 0.8360],
        "XGBoost": [0.7606, 0.7912, 0.8372],
        "CatBoost": [0.7549, 0.7911, 0.8321],
    }
    sig_data = {
        "PyTorch MLP": [381.83, 449.27, 553.31],
        "XGBoost": [376.54, 420.54, 547.65],
        "CatBoost": [368.48, 416.48, 536.39],
    }
    colors = {"PyTorch MLP": "#059669", "XGBoost": "#1D4ED8", "CatBoost": "#D97706"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
    for idx, (model, vals) in enumerate(auc_data.items()):
        bars = ax1.bar(x + (idx - 1) * width, vals, width, label=model,
                       color=colors[model], edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         va="bottom", fontsize=8, fontweight="bold", color="#1E293B")
    ax1.set_ylim(0.70, 0.865)
    ax1.set_xticks(x, experiments, fontsize=9)
    ax1.set_ylabel("Held-Out Test ROC-AUC", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_title("(A) ROC-AUC Progression Across Feature Sets", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    for idx, (model, vals) in enumerate(sig_data.items()):
        bars = ax2.bar(x + (idx - 1) * width, vals, width, label=model,
                       color=colors[model], edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         va="bottom", fontsize=8, fontweight="bold", color="#1E293B")
    ax2.set_ylim(300, 600)
    ax2.set_xticks(x, experiments, fontsize=9)
    ax2.set_ylabel("Discovery Significance Proxy (S / √B)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax2.set_title("(B) Discovery Significance Progression", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    fig.suptitle("HIGGS Feature Ablation Study: 21 Low-Level → 7 High-Level → 28 Combined Features",
                 fontsize=13, fontweight="bold", color="#0F172A", y=1.01)
    fig.tight_layout()
    out_file = OUT_DIR / "02_feature_ablation_progression.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


def plot_feature_importance():
    xgb_path = REPO_ROOT / "xgboost" / "benchmark_1m" / "results" / "experiment_3_5000_rounds_feature_importance.csv"
    cb_path = REPO_ROOT / "catboost" / "benchmark_1m" / "results" / "catboost_experiment_3_28_feature_importance.csv"
    for path in (xgb_path, cb_path):
        if not path.is_file():
            raise FileNotFoundError(f"Feature-importance file not found: {path}")

    xgb_all = pd.read_csv(xgb_path)
    cb_all = pd.read_csv(cb_path)
    required = {"feature", "importance"}
    for label, df in (("XGBoost", xgb_all), ("CatBoost", cb_all)):
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected {label} feature-importance columns: {list(df.columns)}")
        if len(df) != 28:
            raise ValueError(f"Expected 28 {label} feature-importance rows, found {len(df)}")

    xgb_df = xgb_all.sort_values("importance", ascending=False).head(12).copy()
    cb_df = cb_all.sort_values("importance", ascending=False).head(12).copy()
    xgb_df["pct"] = 100.0 * xgb_df["importance"] / xgb_all["importance"].sum()
    cb_df["pct"] = 100.0 * cb_df["importance"] / cb_all["importance"].sum()
    xgb_df = xgb_df.iloc[::-1]
    cb_df = cb_df.iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300)
    xgb_colors = ["#1D4ED8" if f in HIGH_LEVEL_SET else "#94A3B8" for f in xgb_df["feature"]]
    cb_colors = ["#D97706" if f in HIGH_LEVEL_SET else "#94A3B8" for f in cb_df["feature"]]
    xgb_labels = [PHYSICS_NAMES.get(f, f) for f in xgb_df["feature"]]
    cb_labels = [PHYSICS_NAMES.get(f, f) for f in cb_df["feature"]]

    bars1 = ax1.barh(xgb_labels, xgb_df["pct"], color=xgb_colors, edgecolor="white")
    for bar in bars1:
        w = bar.get_width()
        ax1.annotate(f"{w:.1f}%", (w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=8.5, fontweight="bold", color="#1E293B")
    ax1.set_xlim(0, max(xgb_df["pct"]) * 1.18)
    ax1.set_xlabel("Relative Feature Importance (%)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_title("XGBoost (5000 Rounds) — Top 12 Features", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35, color="#CBD5E1")

    bars2 = ax2.barh(cb_labels, cb_df["pct"], color=cb_colors, edgecolor="white")
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.1f}%", (w, bar.get_y() + bar.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=8.5, fontweight="bold", color="#1E293B")
    ax2.set_xlim(0, max(cb_df["pct"]) * 1.18)
    ax2.set_xlabel("Relative Feature Importance (%)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax2.set_title("CatBoost (28 Features) — Top 12 Features", fontsize=12, fontweight="bold", color="#0F172A", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.35, color="#CBD5E1")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1D4ED8", label="High-Level Physics Feature (XGBoost)"),
        Patch(facecolor="#D97706", label="High-Level Physics Feature (CatBoost)"),
        Patch(facecolor="#94A3B8", label="Low-Level Kinematic Feature"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               facecolor="white", edgecolor="#CBD5E1", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Feature Importance: Low-Level Kinematics vs High-Level Physics Features",
                 fontsize=13, fontweight="bold", color="#0F172A", y=1.01)
    fig.tight_layout()
    out_file = OUT_DIR / "03_top_feature_importance_comparison.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    print(f"Repository: {REPO_ROOT}")
    print(f"Loading held-out test split from: {DATA_DIR}")
    X_TEST, Y_TEST = load_test_split()
    print(f"Loaded X_test={X_TEST.shape}, y_test={Y_TEST.shape}")

    model_probabilities = load_all_model_probabilities(X_TEST)
    plot_overlaid_roc(Y_TEST, model_probabilities)
    plot_ablation_progression()
    plot_feature_importance()

    print("\nSUCCESS: All three benchmark plots were generated.")
    for path in sorted(OUT_DIR.glob("0[1-3]_*.png")):
        print(f"{path.name} | {path.stat().st_size / 1024:.1f} KB")
