"""
Unit tests for the Business Decision Layer & Cost Optimization
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.evaluate import (
    compute_business_cost,
    run_threshold_analysis,
    select_optimal_threshold,
)


@pytest.fixture
def synthetic_eval_data():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Predicted probabilities roughly aligned with ground truth
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.35, 0.7, 0.8, 0.85, 0.9])
    return y_true, proba


def test_compute_business_cost(synthetic_eval_data):
    y_true, proba = synthetic_eval_data
    cost_info = compute_business_cost(
        y_true=y_true,
        proba=proba,
        threshold=0.5,
        cost_fn=1200.0,
        cost_fp=50.0,
    )
    assert "total_cost" in cost_info
    assert "cost_per_customer" in cost_info
    assert cost_info["total_cost"] >= 0
    assert cost_info["cost_per_customer"] == cost_info["total_cost"] / len(y_true)


def test_run_threshold_analysis(synthetic_eval_data):
    y_true, proba = synthetic_eval_data
    df_sweep = run_threshold_analysis(
        y_true=y_true,
        proba=proba,
        min_th=0.1,
        max_th=0.9,
        step=0.1,
    )
    assert isinstance(df_sweep, pd.DataFrame)
    assert len(df_sweep) == 9
    assert "total_cost" in df_sweep.columns
    assert "f1" in df_sweep.columns


def test_select_optimal_threshold(synthetic_eval_data):
    y_true, proba = synthetic_eval_data
    df_sweep = run_threshold_analysis(
        y_true=y_true,
        proba=proba,
        min_th=0.1,
        max_th=0.9,
        step=0.05,
    )
    opt_cost_th = select_optimal_threshold(df_sweep, strategy="business_cost")
    assert 0.0 <= opt_cost_th <= 1.0

    opt_f1_th = select_optimal_threshold(df_sweep, strategy="f1")
    assert 0.0 <= opt_f1_th <= 1.0
