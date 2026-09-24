import pandas as pd
import pytest

from src.validation import validate_dataframe


def test_validation_accepts_required_columns():
    df = pd.DataFrame({
        "MedInc": [1.0],
        "HouseAge": [10.0],
        "AveRooms": [4.0],
        "AveBedrms": [1.0],
        "Population": [500.0],
        "AveOccup": [2.0],
        "Latitude": [35.0],
        "Longitude": [-119.0],
        "MedHouseVal": [1.5],
    })

    result = validate_dataframe(df)
    assert result["rows"] == 1
    assert result["missing_values"] == 0


def test_validation_rejects_missing_column():
    df = pd.DataFrame({"MedInc": [1.0]})

    with pytest.raises(ValueError):
        validate_dataframe(df)
