"""
Unit tests for src/pipeline/explain.py (SHAP)
"""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.pipeline.explain import (
    compute_shap_values,
    compute_global_importance,
    explain_single_instance,
)
from src.pipeline.preprocess import build_preprocessing_pipeline, BINARY_COLS, NUMERIC_COLS, OHE_COLS, get_feature_names


@pytest.fixture
def trained_tree_model(raw_df):
    feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
    X_raw = raw_df[feature_cols]
    y = raw_df["Churn"].to_numpy()

    pipe = build_preprocessing_pipeline()
    X_trans = pipe.fit_transform(X_raw)

    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_trans, y)
    return model, pipe


def test_compute_shap_values(trained_tree_model, raw_df):
    model, pipe = trained_tree_model
    feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
    X_trans = pipe.transform(raw_df[feature_cols])

    shap_vals = compute_shap_values(model, X_trans)
    assert shap_vals.shape == X_trans.shape


def test_compute_global_importance(trained_tree_model, raw_df):
    model, pipe = trained_tree_model
    feature_cols = BINARY_COLS + NUMERIC_COLS + OHE_COLS
    X_trans = pipe.transform(raw_df[feature_cols])
    names = get_feature_names(pipe)

    shap_vals = compute_shap_values(model, X_trans)
    df_imp = compute_global_importance(shap_vals, names)

    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == len(names)
    assert "mean_abs_shap" in df_imp.columns
    assert (df_imp["mean_abs_shap"] >= 0).all()


def test_explain_single_instance(trained_tree_model, sample_customer):
    model, pipe = trained_tree_model
    names = get_feature_names(pipe)

    explanation = explain_single_instance(
        model=model,
        pipeline=pipe,
        input_dict=sample_customer,
        feature_names=names,
        top_k=5,
    )

    assert "churn_probability" in explanation
    assert "prediction" in explanation
    assert "top_contributing_factors" in explanation
    assert len(explanation["top_contributing_factors"]) <= 5
