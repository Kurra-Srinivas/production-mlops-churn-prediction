"""
MODEL EVALUATION & BUSINESS DECISION LAYER
===========================================

Computes multi-metric performance, runs threshold sweeps, evaluates expected
financial business costs (False Negatives vs False Positives), and selects
the cost-optimal operating threshold.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full classification metrics at a given threshold.
    """
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_business_cost(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_fn: float = 1200.0,
    cost_fp: float = 50.0,
) -> Dict[str, float]:
    """
    Calculate the total and per-customer financial loss for a classification threshold.

    Cost model:
        Total Cost = (FN * cost_FN) + (FP * cost_FP)
        Cost Per Customer = Total Cost / N
    """
    metrics = evaluate_predictions(y_true, proba, threshold=threshold)
    fn = metrics["fn"]
    fp = metrics["fp"]
    n_samples = len(y_true)

    total_cost = float((fn * cost_fn) + (fp * cost_fp))
    cost_per_customer = float(total_cost / n_samples) if n_samples > 0 else 0.0

    return {
        "threshold": float(threshold),
        "total_cost": total_cost,
        "cost_per_customer": cost_per_customer,
        "fn_cost": float(fn * cost_fn),
        "fp_cost": float(fp * cost_fp),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def run_threshold_analysis(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fn: float = 1200.0,
    cost_fp: float = 50.0,
    min_th: float = 0.01,
    max_th: float = 0.99,
    step: float = 0.001,
) -> pd.DataFrame:
    """
    Sweep classification thresholds to generate precision/recall/cost curves.
    """
    thresholds = np.arange(min_th, max_th + step / 2.0, step)
    records = []

    for th in thresholds:
        record = compute_business_cost(
            y_true=y_true,
            proba=proba,
            threshold=th,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        records.append(record)

    df_results = pd.DataFrame(records)
    return df_results


def select_optimal_threshold(
    threshold_df: pd.DataFrame,
    strategy: str = "business_cost",
) -> float:
    """
    Select the optimal decision threshold from a threshold sweep DataFrame.

    Strategies:
        - 'business_cost': Minimizes total expected financial cost
        - 'f1': Maximizes F1 score
        - 'recall_at_precision': Balance precision/recall
    """
    if strategy == "business_cost":
        idx = threshold_df["total_cost"].idxmin()
    elif strategy == "f1":
        idx = threshold_df["f1"].idxmax()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    optimal_th = float(threshold_df.loc[idx, "threshold"])
    return round(optimal_th, 3)
