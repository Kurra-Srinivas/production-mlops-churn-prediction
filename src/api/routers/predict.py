"""
PREDICTION & EXPLAINABILITY ROUTER
===================================
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CustomerData,
    ExplainResponse,
    FactorExplanation,
    PredictionResponse,
)
from src.pipeline.explain import explain_single_instance
from src.serving.inference import FEATURE_COLS, THRESHOLD, model, pipeline

router = APIRouter(tags=["Prediction & Explainability"])


@router.post("/predict", response_model=PredictionResponse)
def get_prediction(data: CustomerData):
    """
    Predict customer churn outcome and calibrated probability.
    """
    try:
        input_dict = data.model_dump()
        df_raw = pd.DataFrame([input_dict])
        X_trans = pipeline.transform(df_raw)

        raw_model = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else model

        if hasattr(raw_model, "predict_proba"):
            proba = float(raw_model.predict_proba(X_trans)[0, 1])
        else:
            proba = float(raw_model.predict(X_trans)[0])

        prediction_label = "Likely to churn" if proba >= THRESHOLD else "Not likely to churn"

        return PredictionResponse(
            prediction=prediction_label,
            churn_probability=round(proba, 4),
            threshold_used=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e


@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(data: CustomerData):
    """
    Explain customer churn prediction using SHAP feature attribution.
    """
    try:
        input_dict = data.model_dump()
        raw_model = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else model

        result = explain_single_instance(
            model=raw_model,
            pipeline=pipeline,
            input_dict=input_dict,
            feature_names=FEATURE_COLS,
            top_k=5,
        )

        factors = [
            FactorExplanation(
                feature=f["feature"],
                shap_value=f["shap_value"],
                impact=f["impact"],
            )
            for f in result["top_contributing_factors"]
        ]

        return ExplainResponse(
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            base_value=result["base_value"],
            top_contributing_factors=factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}") from e
