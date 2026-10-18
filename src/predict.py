
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "logistic_regression_pipeline.joblib"
)

DEFAULT_THRESHOLD = 0.30


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load a previously trained pipeline from disk."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train and save the model first."
        )

    return joblib.load(model_path)


def predict_churn(
    customer_data,
    model_path=DEFAULT_MODEL_PATH,
    threshold=DEFAULT_THRESHOLD,
):
    """Return churn probabilities and predicted labels."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    elif not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "customer_data must be a dictionary or pandas DataFrame."
        )

    if customer_data.empty:
        raise ValueError("customer_data must contain at least one row.")

    model = load_model(model_path)

    probabilities = model.predict_proba(customer_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": predictions,
            "predicted_label": [
                "Yes" if prediction == 1 else "No"
                for prediction in predictions
            ],
        },
        index=customer_data.index,
    )
