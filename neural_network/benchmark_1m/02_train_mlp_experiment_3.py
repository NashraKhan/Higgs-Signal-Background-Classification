
from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 4096
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")

print("=" * 65)
print("PYTORCH MLP - EXPERIMENT 3 (ALL 28 FEATURES)")
print("=" * 65)
print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Early-stopping patience: {PATIENCE}")

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_dataset(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    y = df["label"].to_numpy(dtype=np.float32)
    X = df[[f"feature_{i}" for i in range(1, 29)]].to_numpy(
        dtype=np.float32
    )

    return X, y


print("\nLoading training dataset...")
X_train, y_train = load_dataset("higgs_train_700k.csv")

print("Loading validation dataset...")
X_val, y_val = load_dataset("higgs_validation_150k.csv")

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# --------------------------------------------------
# 3. Scale features
# Fit ONLY on training data to prevent leakage.
# --------------------------------------------------

print("\nScaling features...")
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

# --------------------------------------------------
# 4. Define the neural network
# --------------------------------------------------

class HiggsMLP(nn.Module):
    def __init__(self, input_features=28):
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


model = HiggsMLP(input_features=28).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

# --------------------------------------------------
# 5. Train with early stopping
# --------------------------------------------------

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_state = None
history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(yb)
        train_count += len(yb)

    train_loss = train_loss_sum / train_count

    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * len(yb)
            val_count += len(yb)

    val_loss = val_loss_sum / val_count
    scheduler.step(val_loss)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.5f} | "
        f"Validation loss: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

training_seconds = time.time() - start_time

if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)
model.eval()

# --------------------------------------------------
# 6. Evaluate on validation data
# --------------------------------------------------

print("\nEvaluating best model on validation data...")

probabilities = []

with torch.no_grad():
    for xb, _ in val_loader:
        logits = model(xb.to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())

probabilities = np.asarray(probabilities)
predictions = (probabilities >= 0.5).astype(np.int32)
y_val_int = y_val.astype(np.int32)

metrics = {
    "experiment": "Experiment 3",
    "model": "PyTorch MLP",
    "features": 28,
    "best_epoch": best_epoch,
    "training_time_seconds": round(training_seconds, 2),
    "validation_loss": round(best_val_loss, 6),
    "accuracy": accuracy_score(y_val_int, predictions),
    "precision": precision_score(
        y_val_int, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val_int, predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_val_int, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_val_int, probabilities),
}

metrics_path = RESULTS_DIR / "mlp_experiment_3_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

history_path = RESULTS_DIR / "mlp_experiment_3_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

cm = confusion_matrix(y_val_int, predictions)

pd.DataFrame(
    cm,
    index=["Actual Background", "Actual Signal"],
    columns=["Predicted Background", "Predicted Signal"],
).to_csv(RESULTS_DIR / "mlp_experiment_3_confusion_matrix.csv")

model_path = MODELS_DIR / "mlp_experiment_3.pt"

torch.save({
    "model_state_dict": model.state_dict(),
    "input_features": 28,
    "best_epoch": best_epoch,
    "seed": SEED,
}, model_path)

# Save scaler parameters for later reproducibility.
np.savez(
    MODELS_DIR / "mlp_experiment_3_scaler.npz",
    mean=scaler.mean_,
    scale=scaler.scale_,
)

print("\n" + "=" * 65)
print("EXPERIMENT 3 RESULTS")
print("=" * 65)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nConfusion matrix:")
print(cm)

print(f"\nMetrics saved: {metrics_path}")
print(f"Training history saved: {history_path}")
print(f"Confusion matrix saved.")
print(f"Model saved: {model_path}")
print("Validation data was used for model selection.")
print("The test dataset was NOT loaded or used.")
print("\nPHASE 3 COMPLETED")
