import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_model(model, X, y, cv=5) -> dict:
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    return {
        "CV_MAE": float(-scores["test_mae"].mean()),
        "CV_RMSE": float(-scores["test_rmse"].mean()),
        "CV_R2": float(scores["test_r2"].mean()),
    }


def metrics_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
