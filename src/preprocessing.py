import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RAW_FEATURES
from .feature_engineering import add_features


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Small sklearn-compatible transformer for feature engineering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        return add_features(X)


def build_preprocessor() -> Pipeline:
    feature_engineer = FeatureEngineerTransformer()

    engineered_columns = RAW_FEATURES + ["RoomsPerBedroom", "BedroomsPerRoom"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, engineered_columns)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("preprocessor", column_transformer),
        ]
    )
