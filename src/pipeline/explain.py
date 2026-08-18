"""
MODEL EXPLAINABILITY (SHAP)
============================

Provides global and local model explainability using TreeExplainer for tree models.
Features:
- Global feature importance (mean absolute SHAP value)
- Local prediction explainability (top positive and negative contributing factors)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import shap


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Construct a SHAP TreeExplainer for an XGBoost model.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values for an array of transformed feature samples.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature, sorted descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return df_importance


def explain_single_instance(
    model: Any,
    pipeline: Any,
    input_dict: Dict[str, Any],
    feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Explain a single customer prediction with their top contributing risk factors.
    """
    df_raw = pd.DataFrame([input_dict])
    X_trans = pipeline.transform(df_raw)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_trans)[0]
    base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    proba = float(model.predict_proba(X_trans)[0, 1])

    # Rank features by absolute impact
    factors = []
    for name, val in zip(feature_names, shap_vals):
        factors.append({
            "feature": name,
            "shap_value": float(round(val, 4)),
            "impact": "increases_churn_risk" if val > 0 else "reduces_churn_risk",
        })

    # Sort by absolute SHAP value
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "churn_probability": round(proba, 4),
        "prediction": "Likely to churn" if proba >= 0.35 else "Not likely to churn",
        "base_value": round(base_val, 4),
        "top_contributing_factors": factors[:top_k],
    }
