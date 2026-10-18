
import numpy as np
import pandas as pd
import pytest

from src.data import load_data
from src.predict import predict_churn
from src.preprocessing import build_preprocessor
from src.train import build_model, prepare_features


def test_preprocessor_builds_successfully():
    preprocessor = build_preprocessor()

    assert preprocessor is not None


def test_preprocessor_produces_finite_features():
    df = load_data()
    X, _ = prepare_features(df)

    transformed = build_preprocessor().fit_transform(X.iloc[:100])

    assert transformed.shape[0] == 100
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


def test_model_contains_preprocessor_and_classifier():
    model = build_model()

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_predict_churn_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        predict_churn(
            customer_data=pd.DataFrame({"tenure": [12]}),
            model_path="nonexistent-model.joblib",
            threshold=1.5,
        )


def test_predict_churn_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="at least one row"):
        predict_churn(
            customer_data=pd.DataFrame(),
            model_path="nonexistent-model.joblib",
        )


def test_predict_churn_rejects_unsupported_input():
    with pytest.raises(TypeError, match="dictionary or pandas DataFrame"):
        predict_churn(
            customer_data="not a customer record",
            model_path="nonexistent-model.joblib",
        )
