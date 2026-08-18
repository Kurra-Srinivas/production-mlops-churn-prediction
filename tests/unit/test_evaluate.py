"""
Unit tests for src/pipeline/evaluate.py
"""

import numpy as np

from src.pipeline.evaluate import evaluate_predictions


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, proba, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2
    assert metrics["fn"] == 0
