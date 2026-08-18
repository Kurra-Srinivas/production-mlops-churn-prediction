"""
PREPROCESSING PIPELINE
=======================

Defines and fits the sklearn ColumnTransformer that transforms raw customer
data into model-ready numeric arrays.

CRITICAL DESIGN PRINCIPLE:
    This module is the single source of truth for ALL feature transformations.
    The SAME fitted Pipeline object is used during:
        - training     (fit_pipeline / transform)
        - evaluation   (transform)
        - inference    (transform via serving/inference.py)

    pd.get_dummies is NOT used anywhere in this pipeline. All encoding is done
    through sklearn transformers so that:
        (a) the transformation is reproducible and serialisable
        (b) train/serve skew is structurally impossible
        (c) the pipeline can be logged as an MLflow artifact and version-controlled

Column taxonomy
---------------
    BINARY_COLS        : 2-category string features → OrdinalEncoder (0 or 1)
    NUMERIC_COLS       : continuous/integer features → passthrough (no scaling needed
                         for tree-based models; StandardScaler added in Phase 2 for LR)
    OHE_COLS           : multi-category string features → OneHotEncoder (drop='first')

The column lists below are the AUTHORITATIVE definitions. They must stay
synchronised with the CustomerData schema in src/app/main.py.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column taxonomy
# ---------------------------------------------------------------------------

# Features that have exactly two categories — encoded as 0 or 1.
# OrdinalEncoder categories are explicit so the mapping is deterministic
# across any data subset (single row or full dataset).
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Numeric features passed through without transformation for tree-based models.
# StandardScaler is applied on top for linear models (Logistic Regression).
NUMERIC_COLS = [
    "SeniorCitizen",  # already 0/1 integer
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Multi-category features one-hot encoded with drop='first' to avoid
# multicollinearity (same behaviour as the previous pd.get_dummies approach).
OHE_COLS = [
    "MultipleLines",  # Yes / No / No phone service
    "InternetService",  # DSL / Fiber optic / No
    "OnlineSecurity",  # Yes / No / No internet service
    "OnlineBackup",  # Yes / No / No internet service
    "DeviceProtection",  # Yes / No / No internet service
    "TechSupport",  # Yes / No / No internet service
    "StreamingTV",  # Yes / No / No internet service
    "StreamingMovies",  # Yes / No / No internet service
    "Contract",  # Month-to-month / One year / Two year
    "PaymentMethod",  # 4 payment methods
]

# Explicit categories for OrdinalEncoder — guarantees the same mapping
# whether fitting on the full dataset or on any subset.
BINARY_CATEGORIES = [
    ["Female", "Male"],  # gender:           Female=0, Male=1
    ["No", "Yes"],  # Partner
    ["No", "Yes"],  # Dependents
    ["No", "Yes"],  # PhoneService
    ["No", "Yes"],  # PaperlessBilling
]


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(scale_numerics: bool = False) -> Pipeline:
    """
    Construct an unfitted sklearn preprocessing Pipeline.

    Args:
        scale_numerics: If True, applies StandardScaler to numeric columns
                        after imputation. Set True for Logistic Regression;
                        leave False (default) for tree-based models.

    Returns:
        An unfitted sklearn Pipeline containing a ColumnTransformer.
        Call .fit_transform(X) or .fit(X) / .transform(X) to use it.

    Column transformer sub-pipelines:
        binary_pipe:  SimpleImputer(most_frequent) + OrdinalEncoder
        numeric_pipe: SimpleImputer(median) [+ StandardScaler if scale_numerics]
        ohe_pipe:     SimpleImputer(most_frequent) + OneHotEncoder(drop=first)
    """
    # Binary sub-pipeline
    binary_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(
                categories=BINARY_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,  # -1 is outside [0, n_categories-1] — no clash
                dtype=np.float64,
            ),
        ),
    ]
    binary_pipe = Pipeline(binary_steps)

    # Numeric sub-pipeline
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numerics:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # One-hot sub-pipeline
    ohe_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(
                drop="first",  # matches previous pd.get_dummies behaviour
                handle_unknown="ignore",  # unknown category → all-zero row (safe)
                sparse_output=False,  # dense array, required by XGBoost
                dtype=np.float64,
            ),
        ),
    ]
    ohe_pipe = Pipeline(ohe_steps)

    # Combine into ColumnTransformer — preserves deterministic column order:
    #   [binary_cols] + [numeric_cols] + [ohe_expanded_cols]
    ct = ColumnTransformer(
        transformers=[
            ("binary", binary_pipe, BINARY_COLS),
            ("numeric", numeric_pipe, NUMERIC_COLS),
            ("ohe", ohe_pipe, OHE_COLS),
        ],
        remainder="drop",  # drop any unexpected columns (safe, explicit)
        verbose_feature_names_out=False,  # clean feature names without prefix
    )

    return Pipeline([("preprocessor", ct)])


# ---------------------------------------------------------------------------
# Fit / transform helpers
# ---------------------------------------------------------------------------


def fit_pipeline(
    df: pd.DataFrame,
    target_col: str = "Churn",
    scale_numerics: bool = False,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Fit the preprocessing pipeline on training data.

    Args:
        df:             DataFrame containing both features and target column.
                        Expected: raw string categoricals (not yet encoded).
        target_col:     Name of the target column.
        scale_numerics: Passed through to build_preprocessing_pipeline().

    Returns:
        (fitted_pipeline, X_transformed, y)
        - fitted_pipeline: sklearn Pipeline fitted on df (save with save_pipeline)
        - X_transformed:   NumPy array ready for model training
        - y:               Target array (same row order as X_transformed)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    pipeline = build_preprocessing_pipeline(scale_numerics=scale_numerics)
    X_transformed = pipeline.fit_transform(X)

    return pipeline, X_transformed, y


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """
    Return the exact ordered list of feature names output by the fitted pipeline.

    This is the equivalent of the feature_columns.txt that was previously
    written manually. The sklearn API derives these from the fitted transformers.

    Args:
        fitted_pipeline: A pipeline that has been fitted via fit_pipeline().

    Returns:
        List of feature name strings in the exact order produced by transform().
    """
    ct = fitted_pipeline.named_steps["preprocessor"]
    return list(ct.get_feature_names_out())


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """
    Serialise the fitted pipeline to disk.

    Args:
        pipeline: A fitted sklearn Pipeline.
        path:     File path for the .pkl artifact (e.g. 'artifacts/pipeline.pkl').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[OK] Pipeline saved to {path}")


def load_pipeline(path: str) -> Pipeline:
    """
    Load a previously fitted pipeline from disk.

    Args:
        path: Path to the .pkl file created by save_pipeline().

    Returns:
        Fitted sklearn Pipeline ready to call .transform() on new data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")
    pipeline = joblib.load(path)
    print(f"[OK] Pipeline loaded from {path}")
    return pipeline
