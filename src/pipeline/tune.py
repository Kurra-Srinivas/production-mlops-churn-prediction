"""
HYPERPARAMETER OPTIMIZATION (OPTUNA + MLFLOW)
==============================================

Wires Optuna hyperparameter optimization directly into the training pipeline
with MLflow nested child runs for full experiment traceability.
"""

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def run_optuna_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
) -> dict[str, Any]:
    """
    Run an Optuna study to find optimal hyperparameters for XGBoost.

    Each trial is logged as a child MLflow run under the active parent run.
    Optimizes for PR-AUC (Average Precision) on the validation set.

    Args:
        X_train: Transformed feature array for training
        y_train: Target labels for training
        X_val: Transformed feature array for validation
        y_val: Target labels for validation
        n_trials: Number of Optuna trials to execute

    Returns:
        Dictionary of best hyperparameters found by Optuna.
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

        has_active_run = mlflow.active_run() is not None
        if has_active_run:
            try:
                with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
                    mlflow.log_params(params)
                    mlflow.set_tag("optuna_trial", trial.number)
                    model = XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    val_proba = model.predict_proba(X_val)[:, 1]
                    val_pr_auc = average_precision_score(y_val, val_proba)
                    mlflow.log_metric("val_pr_auc", val_pr_auc)
                return float(val_pr_auc)
            except Exception:
                pass

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_proba)
        return float(val_pr_auc)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"   Optuna best trial: #{study.best_trial.number}")
    print(f"   Optuna best val PR-AUC: {study.best_value:.4f}")
    print(f"   Optuna best params: {study.best_params}")

    return study.best_params
