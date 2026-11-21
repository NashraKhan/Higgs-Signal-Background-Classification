import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# ==============================================================================
# 1. LOAD VALIDATION & TEST DATA WITH HEADER=NONE & EXACT FEATURE NAMES
# ==============================================================================
DATA_DIR = "../../lightgbm/benchmark_1m/data"
FEATURE_COLS = [f"feature_{i}" for i in range(1, 29)]

X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv", header=None, names=FEATURE_COLS, dtype=np.float32)
y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv", header=None).values.ravel()
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv", header=None, names=FEATURE_COLS, dtype=np.float32)
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv", header=None).values.ravel()

print(f"Loaded Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# ==============================================================================
# 2. METRIC FUNCTIONS: 95% BOOTSTRAP CI & SIGNIFICANCE PROXY (S / sqrt(B))
# ==============================================================================
def evaluate_test_auc_with_ci(y_true, test_probs, n_bootstraps=1000, seed=42):
    y_true = np.asarray(y_true)
    test_probs = np.asarray(test_probs)
    point_auc = roc_auc_score(y_true, test_probs)

    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], test_probs[idx]))

    return point_auc, np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)

def compute_significance_proxy(y_val_true, val_probs, y_test_true, test_probs):
    # Step A: Find optimal cut-off threshold ONLY on Validation Set
    best_thresh = 0.50
    best_val_sig = -1.0
    for thresh in np.linspace(0.10, 0.90, 81):
        val_preds = (val_probs >= thresh).astype(int)
        S_val = np.sum((val_preds == 1) & (y_val_true == 1))
        B_val = np.sum((val_preds == 1) & (y_val_true == 0))
        if B_val > 0 and (S_val / np.sqrt(B_val)) > best_val_sig:
            best_val_sig = S_val / np.sqrt(B_val)
            best_thresh = thresh

    # Step B: Apply chosen threshold to Held-Out Test Set
    test_preds = (test_probs >= best_thresh).astype(int)
    S_test = np.sum((test_preds == 1) & (y_test_true == 1))
    B_test = np.sum((test_preds == 1) & (y_test_true == 0))
    test_sig = S_test / np.sqrt(B_test)
    return best_thresh, test_sig, S_test, B_test

# ==============================================================================
# 3. EVALUATE ALL SAVED MODELS IN benchmark_1m/models/
# ==============================================================================
print("\nSaved models found in models/:")
saved_models = sorted(glob.glob("models/*.json") + glob.glob("models/*.ubj"))
for m in saved_models:
    print(" ->", m)

results = []

for model_path in saved_models:
    fname = os.path.basename(model_path).lower()

    # Automatically select matching columns (21 raw, 7 physics, or all 28)
    if "21" in fname or "experiment_1" in fname:
        label = "Experiment 1 (21 Raw Features)"
        X_v, X_t = X_val.iloc[:, :21], X_test.iloc[:, :21]
    elif "7" in fname or "experiment_2" in fname:
        label = "Experiment 2 (7 Physics Features)"
        X_v, X_t = X_val.iloc[:, 21:28], X_test.iloc[:, 21:28]
    else:
        label = "Experiment 3 (All 28 Features)"
        X_v, X_t = X_val, X_test

    print(f"\nEvaluating {os.path.basename(model_path)} [{label}]...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    val_probs = model.predict_proba(X_v)[:, 1]
    test_probs = model.predict_proba(X_t)[:, 1]

    test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
    auc, ci_low, ci_high = evaluate_test_auc_with_ci(y_test, test_probs)
    thresh, sig, S, B = compute_significance_proxy(y_val, val_probs, y_test, test_probs)

    print(f"  Test Accuracy (0.50 thresh) : {test_acc:.4f}")
    print(f"  Test ROC-AUC                : {auc:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Optimal Val Threshold       : {thresh:.2f}")
    print(f"  Test Significance Proxy     : {sig:.2f} (Signal S={S}, Background B={B})")

    results.append({
        "Model_File": os.path.basename(model_path),
        "Experiment": label,
        "Test_Accuracy": round(test_acc, 4),
        "Test_ROC_AUC": round(auc, 4),
        "CI_95_Low": round(ci_low, 4),
        "CI_95_High": round(ci_high, 4),
        "Chosen_Threshold": round(thresh, 2),
        "Test_Significance_Proxy": round(sig, 2),
        "Test_Signal_S": int(S),
        "Test_Background_B": int(B)
    })

os.makedirs("results", exist_ok=True)
out_csv = "results/15_xgboost_final_test_metrics.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\nDone! Final test metrics saved to: {out_csv}")