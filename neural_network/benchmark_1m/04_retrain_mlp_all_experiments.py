
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
)
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 7
LEARNING_RATE = 0.001

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


def load_dataset(filename, feature_numbers):
    df = pd.read_csv(DATA_DIR / filename)

    columns = [f"feature_{i}" for i in feature_numbers]
    X = df[columns].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)

    return X, y


def train_experiment(experiment_name, feature_numbers):
    print("\n" + "=" * 65)
    print(f"TRAINING {experiment_name.upper()}")
    print(f"Number of features: {len(feature_numbers)}")
    print("=" * 65)

    # Reset random seeds for reproducibility.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_dataset(
        "higgs_train_700k.csv", feature_numbers
    )
    X_val, y_val = load_dataset(
        "higgs_validation_150k.csv", feature_numbers
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # Fit scaler on training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = HiggsMLP(len(feature_numbers))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(yb)
            total_train_rows += len(yb)

        train_loss = total_train_loss / total_train_rows

        model.eval()
        total_val_loss = 0.0
        total_val_rows = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                total_val_loss += loss.item() * len(yb)
                total_val_rows += len(yb)

        val_loss = total_val_loss / total_val_rows
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss:.5f} | "
            f"Validation loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    training_seconds = time.time() - start_time

    if best_state is None:
        raise RuntimeError("No best model was saved.")

    model.load_state_dict(best_state)
    model.eval()

    probabilities = []

    with torch.no_grad():
        for xb, _ in val_loader:
            logits = model(xb)
            probabilities.extend(torch.sigmoid(logits).numpy())

    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(np.int32)
    y_true = y_val.astype(np.int32)

    metrics = {
        "experiment": experiment_name,
        "model": "PyTorch MLP",
        "features": len(feature_numbers),
        "best_epoch": best_epoch,
        "training_time_seconds": round(training_seconds, 2),
        "validation_loss": round(best_val_loss, 6),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / f"mlp_{experiment_name}_metrics.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"mlp_{experiment_name}_history.csv",
        index=False,
    )

    cm = confusion_matrix(y_true, predictions)
    pd.DataFrame(
        cm,
        index=["Actual Background", "Actual Signal"],
        columns=["Predicted Background", "Predicted Signal"],
    ).to_csv(
        RESULTS_DIR / f"mlp_{experiment_name}_confusion_matrix.csv"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_features": len(feature_numbers),
        "feature_numbers": feature_numbers,
        "best_epoch": best_epoch,
        "seed": SEED,
    }, MODELS_DIR / f"mlp_{experiment_name}.pt")

    np.savez(
        MODELS_DIR / f"mlp_{experiment_name}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
    )

    print("\nRESULTS:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(cm)

    return metrics


def main():
    all_metrics = []

    for experiment_name, features in EXPERIMENTS.items():
        metrics = train_experiment(experiment_name, features)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(
        RESULTS_DIR / "mlp_all_experiments_summary.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("ALL MLP EXPERIMENTS COMPLETED")
    print("=" * 65)
    print(summary.to_string(index=False))
    print("\nSummary saved to:")
    print(RESULTS_DIR / "mlp_all_experiments_summary.csv")
    print("\nThe test dataset was not used.")


if __name__ == "__main__":
    main()
