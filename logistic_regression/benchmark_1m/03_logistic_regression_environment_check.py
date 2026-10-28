
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
import joblib

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 65)
    print("HIGGS LOGISTIC REGRESSION - PHASE 3")
    print("LOGISTIC REGRESSION ENVIRONMENT CHECK")
    print("=" * 65)

    # 1. Check environment
    print("\n1. ENVIRONMENT CHECK")
    print("Python version       :", sys.version.split()[0])
    print("NumPy version        :", np.__version__)
    print("SciPy version        :", scipy.__version__)
    print("scikit-learn version :", sklearn.__version__)
    print("Environment check    : PASS")

    # 2. Create a small synthetic dataset
    print("\n2. SYNTHETIC DATASET CHECK")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    print("Feature shape        :", X.shape)
    print("Target shape         :", y.shape)
    print("Dataset creation     : PASS")

    # 3. Train Logistic Regression
    print("\n3. MODEL TRAINING CHECK")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X, y)
    print("Feature scaling      : PASS")
    print("Logistic Regression  : PASS")

    # 4. Predictions and probabilities
    print("\n4. PREDICTION CHECK")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert predictions.shape == (1000,)
    assert probabilities.shape == (1000, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    print("Predictions          : PASS")
    print("Probability outputs  : PASS")

    # 5. ROC-AUC calculation
    print("\n5. ROC-AUC CHECK")
    auc = roc_auc_score(y, probabilities[:, 1])
    print(f"Synthetic ROC-AUC    : {auc:.4f}")
    print("ROC-AUC calculation  : PASS")

    # 6. Save and reload a temporary model
    print("\n6. MODEL SAVE/LOAD CHECK")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.joblib"
        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)

        assert np.array_equal(
            model.predict(X[:20]),
            loaded_model.predict(X[:20]),
        )

    print("Model saving         : PASS")
    print("Model loading        : PASS")

    # 7. Final summary
    print("\n" + "=" * 65)
    print("PHASE 3 RESULT: ALL CHECKS PASSED")
    print("HIGGS CSV files were not used or modified.")
    print("No permanent model file was created.")
    print("=" * 65)


if __name__ == "__main__":
    main()

