# Customer Churn Prediction — Machine Learning Capstone

## Project overview

This project develops a machine-learning workflow to predict customer churn using the IBM Telco Customer Churn dataset. It covers exploratory data analysis, data cleaning, feature preprocessing, model comparison, cross-validation, hyperparameter tuning, classification-threshold analysis, and final evaluation.

The goal is to identify customers who may leave a telecommunications service and explore how churn predictions could support customer-retention strategies.

## Dataset

- **Source:** IBM Telco Customer Churn dataset
- **Original size:** 7,043 customers and 21 columns
- **Target:** `Churn`
- **Model inputs:** 19 features, excluding `customerID` and the target
- **Target distribution:** 73.5% non-churners and 26.5% churners

The notebook and data-loading module expect `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root, alongside this README. The dataset is present in the local project folder but excluded from Git; obtain it separately and follow its applicable terms of use when setting up a fresh clone.

## Data preparation

Exploratory analysis found 11 whitespace-only entries in `TotalCharges`. These were converted to missing values, and the column was converted to a numeric type. The affected rows were retained, with missing numerical values handled by median imputation in the machine-learning pipeline.

The preprocessing workflow uses:

- Median imputation and standardization for numerical features.
- Most-frequent imputation and one-hot encoding for categorical features.
- A `ColumnTransformer` to combine the preprocessing steps.
- Scikit-learn pipelines so preprocessing is fitted within training folds and data leakage is reduced.

The customer identifier is excluded from model features.

## Models and evaluation

The data was split into stratified training and test sets using a random state of 42, with 20% of observations reserved for testing:

- **Training set:** 5,634 observations
- **Test set:** 1,409 observations

The models compared using five-fold stratified cross-validation were a majority-class dummy classifier, logistic regression, a decision tree, and a random forest. Logistic regression achieved the strongest mean cross-validation ROC-AUC among the three trained predictive models in the initial comparison.

Grid search selected `C = 10` for logistic regression, with a mean cross-validation ROC-AUC of 0.8463.

### Final test results

The selected logistic-regression pipeline was evaluated on the held-out test set with a classification threshold of 0.30.

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.7488 |
| Precision | 0.5185 |
| Recall | 0.7513 |
| F1-score | 0.6135 |
| ROC-AUC | 0.8412 |

The threshold of 0.30 was selected provisionally because it produced the highest F1-score among six thresholds evaluated using out-of-fold training predictions. It prioritizes identifying more potential churners at the expense of additional false-positive predictions. The appropriate business threshold depends on the costs and benefits of retention actions.

## Exploratory findings

The exploratory analysis found descriptive associations between churn and several customer characteristics:

- Customers who churned tended to have shorter tenures.
- Month-to-month contracts had a higher observed churn rate than one-year and two-year contracts.
- Fiber-optic customers had a higher observed churn rate than other internet-service groups.
- Customers using electronic checks had a higher observed churn rate than customers using other payment methods.

These findings do not establish causation.

## Model interpretation

The logistic-regression coefficients and corresponding odds ratios were examined to understand the model's learned associations. Numerical features were standardized, so their coefficients correspond to changes in standardized values. Categorical coefficients depend on the reference categories used by one-hot encoding. These coefficients describe conditional model associations, not causal effects.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # local dataset; excluded from Git
├── models/
│   └── logistic_regression_pipeline.joblib  # local trained model; excluded from Git
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # analysis, model selection, and evaluation
├── src/
│   ├── __init__.py
│   ├── data.py             # data loading and validation
│   ├── preprocessing.py    # feature preprocessing pipeline
│   ├── train.py            # model construction and training
│   └── predict.py          # saved-model loading and churn predictions
└── tests/
    ├── test_data.py
    └── test_model.py
```

The dataset and trained model are ignored by Git, so they must be supplied or generated locally. `train.py` saves the model to `models/logistic_regression_pipeline.joblib` by default.

## Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter, and pytest.

## Reproducibility

The notebook was developed in a Python virtual environment. Dependencies are pinned in `requirements.txt`. Fixed random seeds are used for the train-test split, cross-validation, and applicable model estimators.

## Installation and usage

### Prerequisites

- Python and Git
- The project repository cloned locally
- The Telco Customer Churn dataset in the project root (download separately)

Open PowerShell in the project root and run:

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Register the Jupyter kernel

```powershell
python -m ipykernel install --user --name customer-churn-ml --display-name "Python (Customer Churn ML)"
```

### 4. Add the dataset

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root, alongside this README.

### 5. Run the notebook

Open `notebooks/01_exploratory_data_analysis.ipynb` in VS Code, select the `Python (Customer Churn ML)` kernel, and run the notebook cells in order.

### Train and use the model from Python

The reusable modules can train the pipeline and save it to the default model path:

```powershell
python -c "from src.train import train_from_csv; train_from_csv()"
```

To predict churn for a row from the dataset using the saved pipeline:

```powershell
python -c "from src.data import load_data; from src.predict import predict_churn; df = load_data(); print(predict_churn(df.drop(columns=['Churn', 'customerID']).iloc[[0]]))"
```

`predict_churn` accepts a dictionary for one customer or a pandas DataFrame with the model feature columns. It returns the churn probability, a binary prediction, and a `Yes`/`No` label. The default classification threshold is `0.30`; it can be overridden with the `threshold` argument.

### Run tests

From the project root, run:

```powershell
python -m pytest
```

The tests load the CSV from the project root, so the dataset must be present.

## Future improvements

- Select the classification threshold using an explicit business-cost objective.
- Assess probability calibration and validate performance on newer or independent data.
- Evaluate additional models and class-imbalance strategies where justified.

## Limitations

Model performance is dataset-dependent and does not guarantee future business outcomes. Threshold selection involves trade-offs between missed churners and false-positive alerts. Further validation would be required before deploying the model in a real customer-retention workflow.
