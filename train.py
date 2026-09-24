import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    DISPLAY_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_california_housing
from src.evaluation import cross_validate_model, metrics_dataframe, regression_metrics
from src.explainability import permutation_feature_importance
from src.modeling import get_models, get_tuning_space
from src.preprocessing import build_preprocessor
from src.validation import validate_dataframe

warnings.filterwarnings("ignore", category=UserWarning)


def save_basic_eda(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True)
    plt.title("Target Distribution — Median House Value")
    plt.xlabel("Median House Value ($100,000 units)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_california_housing(save_csv=True)

    validation = validate_dataframe(df)
    print("Validation:", validation)

    save_basic_eda(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    fitted_models = {}

    print("\nTraining baseline models...")
    for name, estimator in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor()),
                ("model", estimator),
            ]
        )

        cv_result = cross_validate_model(pipeline, X_train, y_train, cv=CV_FOLDS)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        test_result = regression_metrics(y_test, predictions)

        row = {"Model": name, **cv_result, **test_result}
        results.append(row)
        fitted_models[name] = pipeline

        print(
            f"{name}: RMSE={test_result['RMSE']:.4f}, "
            f"MAE={test_result['MAE']:.4f}, R2={test_result['R2']:.4f}"
        )

    print("\nTuning Random Forest...")
    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", get_models()["Random Forest"]),
        ]
    )

    search = RandomizedSearchCV(
        estimator=tuned_pipeline,
        param_distributions=get_tuning_space(),
        n_iter=12,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    tuned_predictions = tuned_model.predict(X_test)
    tuned_test = regression_metrics(y_test, tuned_predictions)
    tuned_cv = cross_validate_model(tuned_model, X_train, y_train, cv=CV_FOLDS)

    tuned_row = {
        "Model": "Tuned Random Forest",
        **tuned_cv,
        **tuned_test,
    }
    results.append(tuned_row)
    fitted_models["Tuned Random Forest"] = tuned_model

    print(
        f"Tuned Random Forest: RMSE={tuned_test['RMSE']:.4f}, "
        f"MAE={tuned_test['MAE']:.4f}, R2={tuned_test['R2']:.4f}"
    )

    metrics_df = metrics_dataframe(results)
    metrics_df.to_csv(METRICS_DIR / "model_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="RMSE", y="Model")
    plt.title("Model Comparison — RMSE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Select the model with lowest holdout RMSE.
    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)

    # Save residual plot.
    residuals = y_test.to_numpy() - best_predictions
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=best_predictions, y=residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residual Plot — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=160)
    plt.close()

    importance = permutation_feature_importance(
        best_model, X_test, y_test, random_state=RANDOM_STATE
    )
    importance["DisplayFeature"] = importance["Feature"].map(
        lambda x: DISPLAY_NAMES.get(x, x)
    )
    importance.to_csv(METRICS_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plot_data = importance.head(10).sort_values("Importance")
    sns.barplot(data=plot_data, x="Importance", y="DisplayFeature")
    plt.title(f"Permutation Importance — {best_name}")
    plt.xlabel("Mean decrease in model score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=160)
    plt.close()

    dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "best_params": search.best_params_,
        "metrics": metrics_df.to_dict(orient="records"),
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "engineered_features": [
            "RoomsPerBedroom",
            "BedroomsPerRoom",
        ],
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining completed.")
    print(f"Best model: {best_name}")
    print(metrics_df.to_string(index=False))
    print("\nArtifacts saved under models/ and outputs/.")


if __name__ == "__main__":
    main()
