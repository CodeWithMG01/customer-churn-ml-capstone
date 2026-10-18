Customer Churn Prediction — Machine Learning Capstone
Project Overview

This project develops an end-to-end machine learning workflow to predict customer churn using the IBM Telco Customer Churn dataset.

The goal is to identify customers who are likely to leave a telecommunications service and investigate how machine learning can support customer-retention strategies.

The project covers exploratory data analysis, data cleaning, feature preprocessing, model comparison, cross-validation, hyperparameter tuning, classification-threshold analysis, and final evaluation.

Dataset
Source: IBM Telco Customer Churn dataset
Original observations: 7,043 customers
Original columns: 21
Target: Churn
Model input features: 19, excluding customerID and the target
Target distribution: 73.5% non-churners and 26.5% churners

The dataset file is excluded from Git tracking. Consult the original dataset source and its applicable terms for access and usage information.

Data Preparation

The exploratory analysis identified 11 whitespace-only entries in TotalCharges. These were converted to missing values, and the column was converted to a numeric data type.

The affected rows were retained. Missing numerical values are handled through median imputation inside the machine learning pipeline.

The feature preprocessing workflow includes:

Median imputation and standardization for numerical features.
Most-frequent imputation and one-hot encoding for categorical features.
ColumnTransformer to combine the preprocessing steps.
Scikit-learn pipelines to keep preprocessing fitted within training folds and reduce data leakage.

The customer identifier is excluded from the model features.

Models and Evaluation

The data was divided into stratified training and test sets:

Training set: 5,634 observations
Test set: 1,409 observations
Random state: 42
Test proportion: 20%

The models compared using five-fold stratified cross-validation were:

Majority-class dummy classifier
Logistic regression
Decision tree
Random forest

Logistic regression achieved the strongest mean cross-validation ROC-AUC among the three trained predictive models in the initial comparison.

Hyperparameter tuning with GridSearchCV selected C = 10 for logistic regression, with a mean cross-validation ROC-AUC of 0.8463.

Final Test Results

The selected logistic regression pipeline was evaluated on the held-out test set using a classification threshold of 0.30.

Metric	Test result
Accuracy	0.7488
Precision	0.5185
Recall	0.7513
F1-score	0.6135
ROC-AUC	0.8412

The threshold of 0.30 was selected provisionally because it produced the highest F1-score among six thresholds evaluated using out-of-fold training predictions.

The selected threshold prioritizes identifying more potential churners at the expense of additional false-positive predictions. The appropriate business threshold would ultimately depend on the costs and benefits of customer-retention actions.

Key Exploratory Findings

The exploratory analysis identified several associations with churn:

Customers who churned tended to have shorter tenures.
Month-to-month contracts had a higher observed churn rate than one-year and two-year contracts.
Fiber-optic customers had a higher observed churn rate than other internet-service groups.
Customers using electronic checks had a higher observed churn rate than customers using other payment methods.

These findings are descriptive associations and do not establish causation.

Model Interpretation

The logistic regression coefficients and corresponding odds ratios were examined to understand the model's learned associations.

Numerical features were standardized, so their coefficients correspond to changes in standardized values. Categorical coefficients depend on the reference categories used by one-hot encoding.

The coefficients describe conditional model associations, not causal effects.

Project Structure
Machine learning project/
├── .gitignore
├── README.md
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
└── WA_Fn-UseC_-Telco-Customer-Churn.csv


The dataset CSV is kept locally and excluded from Git tracking.

Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Git and GitHub
Reproducibility

The notebook was developed in a Python virtual environment. The modeling workflow uses fixed random seeds for the train-test split, cross-validation, and applicable model estimators.

A dependency file and final execution instructions will be added as the project is prepared for publication.

Future Improvements
Evaluate additional models and tune their hyperparameters.
Investigate class-imbalance strategies where justified.
Refine threshold selection using an explicit business-cost objective.
Assess model calibration and interpretability.
Refactor reusable modeling code into Python modules.
Add a reproducible dependency file and usage instructions.
Publish the repository on GitHub.
Limitations

Model performance is dataset-dependent and does not guarantee future business outcomes. Threshold selection involves trade-offs between missed churners and false-positive alerts. Further validation would be required before deploying the model in a real customer-retention workflow.