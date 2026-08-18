"""
MODEL TRAINING
==============

Provides factory and training functions for the baseline and candidate models:
1. Logistic Regression (Linear interpretable baseline)
2. Random Forest (Non-linear bagging ensemble baseline)
3. XGBoost (Gradient boosted decision tree model)
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model with balanced class weighting.
    """
    default_params: Dict[str, Any] = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> RandomForestClassifier:
    """
    Train a Random Forest ensemble baseline model.
    """
    default_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> XGBClassifier:
    """
    Train an XGBoost classifier with dynamic positive class weighting.
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0

    default_params: Dict[str, Any] = {
        "n_estimators": 301,
        "learning_rate": 0.034,
        "max_depth": 7,
        "subsample": 0.95,
        "colsample_bytree": 0.98,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model
