"""
PYDANTIC DATA SCHEMAS
======================

Strict Pydantic v2 data models for API input validation and structured responses.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerData(BaseModel):
    """
    Input schema representing a single customer with all 19 required features.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # Demographics
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # Account info
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0)
    TotalCharges: float = Field(..., ge=0.0, le=15000.0)

    # Phone Services
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    # Internet Services
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]


class PredictionResponse(BaseModel):
    """
    Prediction output schema with classification outcome and probability.
    """

    prediction: Literal["Likely to churn", "Not likely to churn"]
    churn_probability: float
    threshold_used: float


class FactorExplanation(BaseModel):
    feature: str
    shap_value: float
    impact: str


class ExplainResponse(BaseModel):
    """
    Prediction explanation output schema with SHAP attributions.
    """

    prediction: Literal["Likely to churn", "Not likely to churn"]
    churn_probability: float
    base_value: float
    top_contributing_factors: list[FactorExplanation]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool
