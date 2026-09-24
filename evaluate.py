"""Convenience entry point for evaluating an already-trained model."""

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_california_housing
from src.evaluation import regression_metrics


def main():
    model = load(MODELS_DIR / "best_model.joblib")
    df = load_california_housing(save_csv=False)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
