import numpy as np
import pandas as pd

from .config import RAW_FEATURES, TARGET_COLUMN


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Return validation information and raise only for structural errors."""
    required = RAW_FEATURES + [TARGET_COLUMN]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_columns = required
    non_numeric = [
        c for c in numeric_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric columns found: {non_numeric}")

    inf_counts = {
        c: int(np.isinf(df[c].to_numpy()).sum())
        for c in numeric_columns
    }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df[required].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "infinite_values": int(sum(inf_counts.values())),
        "infinite_by_column": inf_counts,
    }
