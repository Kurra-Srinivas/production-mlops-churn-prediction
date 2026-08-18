"""
Unit tests for src/pipeline/train.py and src/pipeline/tune.py
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.pipeline.train import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.pipeline.tune import run_optuna_study


@pytest.fixture
def synthetic_train_val():
    np.random.seed(42)
    X_train = np.random.randn(80, 30)
    y_train = np.random.choice([0, 1], size=80, p=[0.7, 0.3])
    X_val = np.random.randn(20, 30)
    y_val = np.random.choice([0, 1], size=20, p=[0.7, 0.3])
    return X_train, y_train, X_val, y_val


def test_train_logistic_regression(synthetic_train_val):
    X_train, y_train, _, _ = synthetic_train_val
    model = train_logistic_regression(X_train, y_train)
    assert isinstance(model, LogisticRegression)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_train_random_forest(synthetic_train_val):
    X_train, y_train, _, _ = synthetic_train_val
    model = train_random_forest(X_train, y_train, params={"n_estimators": 10})
    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_train_xgboost(synthetic_train_val):
    X_train, y_train, _, _ = synthetic_train_val
    model = train_xgboost(X_train, y_train, params={"n_estimators": 10})
    assert isinstance(model, XGBClassifier)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_optuna_study_returns_valid_params(synthetic_train_val):
    X_train, y_train, X_val, y_val = synthetic_train_val
    best_params = run_optuna_study(X_train, y_train, X_val, y_val, n_trials=3)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params
    assert "max_depth" in best_params
