"""
Shared pytest fixtures for the Telco Churn test suite.
"""

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import pytest

from src.pipeline.preprocess import (
    BINARY_COLS,
    NUMERIC_COLS,
    OHE_COLS,
    build_preprocessing_pipeline,
)

_RAW_ROWS = [
    {
        "customerID": "0001-ABCD1",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
        "Churn": 1,
    },
    {
        "customerID": "0002-ABCD2",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "Churn": 0,
    },
    {
        "customerID": "0003-ABCD3",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": 240.0,
        "Churn": 0,
    },
    {
        "customerID": "0004-ABCD4",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
        "Churn": 1,
    },
]


def _make_synthetic_df(n_samples: int = 20) -> pd.DataFrame:
    rows = [_RAW_ROWS[i % len(_RAW_ROWS)] for i in range(n_samples)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return _make_synthetic_df(n_samples=20)


@pytest.fixture(scope="session")
def sample_customer() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    }


@pytest.fixture(scope="session")
def fitted_pipeline(raw_df):
    feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
    X = raw_df[feature_cols]
    pipeline = build_preprocessing_pipeline(scale_numerics=False)
    pipeline.fit(X)
    return pipeline


@pytest.fixture(scope="session")
def fitted_pipeline_scaled(raw_df):
    feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
    X = raw_df[feature_cols]
    pipeline = build_preprocessing_pipeline(scale_numerics=True)
    pipeline.fit(X)
    return pipeline
