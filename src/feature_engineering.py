import numpy as np
import pandas as pd

from .config import ENGINEERED_FEATURES


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create robust domain-inspired ratios while protecting against division by zero."""
    X = X.copy()

    rooms = X["AveRooms"].clip(lower=1e-6)
    bedrooms = X["AveBedrms"].clip(lower=1e-6)

    X["RoomsPerBedroom"] = rooms / bedrooms
    X["BedroomsPerRoom"] = bedrooms / rooms

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def engineered_feature_names() -> list[str]:
    return ENGINEERED_FEATURES.copy()
