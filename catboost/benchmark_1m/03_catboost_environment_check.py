
from pathlib import Path
import tempfile

import catboost
import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

def main():
    print("CATBOOST ENVIRONMENT CHECK")
    print(f"CatBoost version: {catboost.__version__}")

    # Generate a small synthetic dataset; no HIGGS files are used.
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    model = CatBoostClassifier(
        iterations=30,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
    )

    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probabilities)

    assert probabilities.shape == (1000,)
    assert np.isfinite(probabilities).all()

    print("[PASS] Model trained")
    print(f"[PASS] Probability output shape: {probabilities.shape}")
    print(f"[PASS] Synthetic-data ROC-AUC: {auc:.4f}")

    # Verify model saving and loading in a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "catboost_test.cbm"
        model.save_model(str(model_path))

        loaded_model = CatBoostClassifier()
        loaded_model.load_model(str(model_path))

        loaded_probabilities = loaded_model.predict_proba(X)[:, 1]

        if not np.allclose(probabilities, loaded_probabilities):
            raise RuntimeError("Predictions differ after reloading the model.")

        print("[PASS] Model saved and loaded successfully")
        print("[PASS] Predictions match after reloading")

    print("\nSUCCESS: CatBoost environment is ready.")
    print("The HIGGS dataset was not accessed.")

if __name__ == "__main__":
    main()
