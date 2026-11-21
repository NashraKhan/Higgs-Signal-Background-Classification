from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_DIR = Path("../../lightgbm/benchmark_1m/data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 7
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "experiment_1": list(range(1, 22)),
    "experiment_2": list(range(22, 29)),
    "experiment_3": list(range(1, 29)),
}


class HiggsMLP(nn.Module):
    def __init__(self, input_features):
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


def get_probabilities(model, X_scaled):
    dataset = TensorDataset(torch.from_numpy(X_scaled))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.asarray(probs)


def evaluate_test_auc_with_ci(y_true, test_probs, n_bootstraps=1000, seed=SEED):
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
    best_thresh, best_val_sig = 0.50, -1.0
    for thresh in np.linspace(0.10, 0.90, 81):
        val_preds = (val_probs >= thresh).astype(int)
        S_val = np.sum((val_preds == 1) & (y_val_true == 1))
        B_val = np.sum((val_preds == 1) & (y_val_true == 0))
        if B_val > 0 and (S_val / np.sqrt(B_val)) > best_val_sig:
            best_val_sig = S_val / np.sqrt(B_val)
            best_thresh = thresh

    test_preds = (test_probs >= best_thresh).astype(int)
    S_test = np.sum((test_preds == 1) & (y_test_true == 1))
    B_test = np.sum((test_preds == 1) & (y_test_true == 0))
    return best_thresh, S_test / np.sqrt(B_test), S_test, B_test


def main():
    print(f"Using device: {DEVICE}")
    print("Loading train, validation, and test datasets from lightgbm/benchmark_1m/data/...")

    X_train_full = pd.read_csv(DATA_DIR / "X_train.csv", header=None, dtype=np.float32).to_numpy()
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", header=None, dtype=np.float32).to_numpy().ravel()

    X_val_full = pd.read_csv(DATA_DIR / "X_val.csv", header=None, dtype=np.float32).to_numpy()
    y_val = pd.read_csv(DATA_DIR / "y_val.csv", header=None, dtype=np.float32).to_numpy().ravel()

    X_test_full = pd.read_csv(DATA_DIR / "X_test.csv", header=None, dtype=np.float32).to_numpy()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv", header=None, dtype=np.float32).to_numpy().ravel()

    test_results = []

    for exp_name, feature_numbers in EXPERIMENTS.items():
        print("\n" + "=" * 65)
        print(f"TRAINING & HELD-OUT TEST EVALUATION: {exp_name.upper()} ({len(feature_numbers)} features)")
        print("=" * 65)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        col_idx = [f - 1 for f in feature_numbers]
        X_train = X_train_full[:, col_idx]
        X_val = X_val_full[:, col_idx]
        X_test = X_test_full[:, col_idx]

        # Fit StandardScaler strictly on Training Set
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val)),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = HiggsMLP(len(feature_numbers)).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improvement = 0
        start_time = time.time()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            total_train_loss, total_train_rows = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(yb)
                total_train_rows += len(yb)
            train_loss = total_train_loss / total_train_rows

            model.eval()
            total_val_loss, total_val_rows = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = criterion(model(xb), yb)
                    total_val_loss += loss.item() * len(yb)
                    total_val_rows += len(yb)
            val_loss = total_val_loss / total_val_rows
            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                break

        train_time = round(time.time() - start_time, 2)
        model.load_state_dict(best_state)

        # Save .pt checkpoint and scaler
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_features": len(feature_numbers),
            "feature_numbers": feature_numbers,
            "best_epoch": best_epoch,
            "seed": SEED,
        }, MODELS_DIR / f"mlp_{exp_name}.pt")

        np.savez(
            MODELS_DIR / f"mlp_{exp_name}_scaler.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
        )

        # Evaluate on Validation and Held-Out Test Set
        val_probs = get_probabilities(model, X_val_scaled)
        test_probs = get_probabilities(model, X_test_scaled)
        y_val_int = y_val.astype(np.int32)
        y_test_int = y_test.astype(np.int32)
        test_preds_05 = (test_probs >= 0.5).astype(np.int32)

        test_auc, ci_low, ci_high = evaluate_test_auc_with_ci(y_test_int, test_probs)
        best_thresh, test_sig, S_test, B_test = compute_significance_proxy(
            y_val_int, val_probs, y_test_int, test_probs
        )

        row = {
            "experiment": exp_name,
            "features": len(feature_numbers),
            "best_epoch": best_epoch,
            "train_time_sec": train_time,
            "test_accuracy": round(accuracy_score(y_test_int, test_preds_05), 4),
            "test_precision": round(precision_score(y_test_int, test_preds_05, zero_division=0), 4),
            "test_recall": round(recall_score(y_test_int, test_preds_05, zero_division=0), 4),
            "test_f1": round(f1_score(y_test_int, test_preds_05, zero_division=0), 4),
            "test_roc_auc": round(test_auc, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "optimal_val_threshold": round(best_thresh, 2),
            "test_significance_proxy": round(test_sig, 2),
            "test_signal_S": int(S_test),
            "test_background_B": int(B_test),
        }
        test_results.append(row)

        for k, v in row.items():
            print(f"  {k}: {v}")

        cm = confusion_matrix(y_test_int, test_preds_05)
        pd.DataFrame(
            cm,
            index=["Actual Background", "Actual Signal"],
            columns=["Predicted Background", "Predicted Signal"],
        ).to_csv(RESULTS_DIR / f"mlp_{exp_name}_test_confusion_matrix.csv")

    summary_df = pd.DataFrame(test_results)
    summary_df.to_csv(RESULTS_DIR / "mlp_final_test_results.csv", index=False)

    print("\n" + "=" * 65)
    print("FINAL PYTORCH MLP HELD-OUT TEST SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {RESULTS_DIR / 'mlp_final_test_results.csv'}")


if __name__ == "__main__":
    main()