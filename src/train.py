
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import ID_COLUMN, TARGET_COLUMN, load_data, validate_data
from src.preprocessing import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "logistic_regression_pipeline.joblib"
)


def build_model(C=10):
    """Build the preprocessing and logistic regression pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(X_train, y_train, C=10, model_path=None):
    """Fit a model using training data and optionally save it."""

    model = build_model(C=C)
    model.fit(X_train, y_train)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return model


def prepare_features(df):
    """Separate the target and exclude the customer identifier."""

    validate_data(df)

    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"].astype(str).str.strip().replace("", float("nan")),
        errors="raise",
    )

    return X, y


def train_from_csv(data_path=None, C=10, model_path=DEFAULT_MODEL_PATH):
    """Load data, create a stratified split, train, and save the model."""

    from sklearn.model_selection import train_test_split

    df = load_data() if data_path is None else load_data(data_path)
    X, y = prepare_features(df)

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = train_model(
        X_train,
        y_train,
        C=C,
        model_path=model_path,
    )

    return model
