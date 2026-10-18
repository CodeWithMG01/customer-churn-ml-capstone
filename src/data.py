
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_data(data_path=DEFAULT_DATA_PATH):
    """Load the Telco Customer Churn dataset."""

    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Download the dataset and place it in the project root."
        )

    return pd.read_csv(data_path)


def validate_data(df):
    """Validate the basic structure of the churn dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[ID_COLUMN].isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Customer IDs must be unique.")

    if not df[TARGET_COLUMN].isin(["Yes", "No"]).all():
        raise ValueError(
            "Churn must contain only 'Yes' and 'No' values."
        )

    return True
