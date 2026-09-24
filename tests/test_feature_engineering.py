import pandas as pd

from src.feature_engineering import add_features


def test_feature_engineering_adds_expected_columns():
    X = pd.DataFrame({
        "MedInc": [3.0],
        "HouseAge": [20.0],
        "AveRooms": [5.0],
        "AveBedrms": [1.0],
        "Population": [1000.0],
        "AveOccup": [3.0],
        "Latitude": [35.0],
        "Longitude": [-119.0],
    })

    result = add_features(X)

    assert "RoomsPerBedroom" in result.columns
    assert "BedroomsPerRoom" in result.columns
    assert result.loc[0, "RoomsPerBedroom"] == 5.0
