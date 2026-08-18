"""
TELCO CUSTOMER CHURN — PRODUCTION API & UI
===========================================

FastAPI application integrating modular REST routers for prediction, explainability,
health checks, and an interactive Gradio UI.
"""

import os
from fastapi import FastAPI
import gradio as gr

from src.api.routers import predict, health
from src.serving.inference import predict as model_predict

# ---------------------------------------------------------------------------
# FastAPI Application Configuration
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telco Customer Churn Prediction & Explainability API",
    description=(
        "Production-grade ML API for telecom customer churn prediction. "
        "Provides inference (/predict), SHAP explainability (/explain), and health status (/health)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount Routers
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "title": "Telco Customer Churn API",
        "status": "active",
        "endpoints": ["/predict", "/explain", "/health", "/ready", "/ui", "/docs"],
    }


# ---------------------------------------------------------------------------
# Gradio UI Integration
# ---------------------------------------------------------------------------

def gradio_predict(
    gender, SeniorCitizen, Partner, Dependents,
    tenure, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges,
    PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
):
    data = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
    }
    try:
        return model_predict(data)
    except Exception as e:
        return f"Error: {e}"


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Slider(minimum=0, maximum=1, step=1, label="Senior Citizen (0=No, 1=Yes)", value=0),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=120),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method", value="Electronic check",
        ),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="Predict churn probabilities and risk levels using an optimized gradient boosting model.",
    theme=gr.themes.Soft(),
)

app = gr.mount_gradio_app(app, demo, path="/ui")
