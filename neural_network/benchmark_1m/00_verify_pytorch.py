import sys
import torch
import pandas as pd
import numpy as np
import sklearn

print("=" * 70)
print("PYTORCH MLP - ENVIRONMENT VERIFICATION")
print("=" * 70)

print("\nPython version:")
print(sys.version)

print("\nInstalled library versions:")
print("-" * 40)
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Pandas        : {pd.__version__}")
print(f"NumPy         : {np.__version__}")
print(f"Scikit-learn  : {sklearn.__version__}")

print("\nDevice information:")
print("-" * 40)

if torch.cuda.is_available():
    print("GPU available: YES")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU available: NO")
    print("The MLP can run on the CPU.")
    device = torch.device("cpu")

print(f"Selected device: {device}")

print("\nRunning a small neural network test...")
print("-" * 40)

# Create a small example neural network.
model = torch.nn.Sequential(
    torch.nn.Linear(28, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)

model = model.to(device)

# Generate artificial data for a basic functionality test.
sample_input = torch.randn(16, 28, device=device)

with torch.no_grad():
    output = model(sample_input)

print(f"Input shape : {tuple(sample_input.shape)}")
print(f"Output shape: {tuple(output.shape)}")

if output.shape == (16, 1):
    print("\nPASS: PyTorch neural network executed successfully.")
else:
    print("\nFAIL: Unexpected output shape.")

print("\n" + "=" * 70)
print("ENVIRONMENT VERIFICATION COMPLETED")
print("=" * 70)