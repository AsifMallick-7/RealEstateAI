from pathlib import Path

import pandas as pd
from joblib import load

from src.config import MODELS_DIR


def load_trained_model(path: str | Path = MODELS_DIR / "best_model.joblib"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` first."
        )
    return load(path)


def predict_price(model, values: dict) -> float:
    row = pd.DataFrame([values])
    prediction = float(model.predict(row)[0])
    return prediction
