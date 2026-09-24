import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_feature_importance(
    fitted_pipeline, X_test: pd.DataFrame, y_test, random_state: int = 42
) -> pd.DataFrame:
    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False)

    return importance.reset_index(drop=True)
