
import pandas as pd
import pytest

from src.data import load_data, validate_data
from src.train import prepare_features


def test_load_data_has_expected_shape():
    df = load_data()

    assert df.shape == (7043, 21)


def test_validate_data_accepts_original_dataset():
    df = load_data()

    assert validate_data(df) is True


def test_validate_data_rejects_duplicate_customer_ids():
    df = load_data()
    df.loc[df.index[1], "customerID"] = df.loc[df.index[0], "customerID"]

    with pytest.raises(ValueError, match="unique"):
        validate_data(df)


def test_validate_data_rejects_invalid_churn_label():
    df = load_data()
    df.loc[df.index[0], "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Churn"):
        validate_data(df)


def test_prepare_features_excludes_id_and_target():
    df = load_data()

    X, y = prepare_features(df)

    assert X.shape == (7043, 19)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y.name == "Churn"


def test_prepare_features_preserves_missing_total_charges():
    df = load_data()

    X, _ = prepare_features(df)

    assert X["TotalCharges"].isna().sum() == 11
