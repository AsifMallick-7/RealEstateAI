from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

from .config import TARGET_COLUMN, RAW_FEATURES, RAW_DATA_DIR


def load_california_housing(save_csv: bool = True) -> pd.DataFrame:
    """Download/load the California Housing dataset and return a DataFrame."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df.columns = list(dataset.feature_names) + [TARGET_COLUMN]

    if save_csv:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_DIR / "california_housing.csv", index=False)

    return df


def load_local_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = set(RAW_FEATURES + [TARGET_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df
