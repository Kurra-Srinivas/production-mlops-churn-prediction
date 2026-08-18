# 🔮 Production Telecom Customer Churn Prediction & MLOps Platform

[![CI Pipeline](https://github.com/your-username/Telco-Customer-Churn-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Telco-Customer-Churn-ML/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.15+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end, portfolio-grade Machine Learning and MLOps system engineered for reproducible data validation, leakage-free preprocessing, experiment tracking, hyperparameter optimization, business-cost threshold optimization, SHAP explainability, and containerized API serving.**

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture](#-system-architecture)
- [Key Engineering Highlights](#-key-engineering-highlights)
- [Data Pipeline & Zero-Skew Guarantee](#-data-pipeline--zero-skew-guarantee)
- [Model Exploration & Benchmarking](#-model-exploration--benchmarking)
- [Business Decision Layer & Cost Optimization](#-business-decision-layer--cost-optimization)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [FastAPI & Gradio Serving](#-fastapi--gradio-serving)
- [CI/CD & Containerization](#-cicd--containerization)
- [Getting Started & Local Reproduction](#-getting-started--local-reproduction)
- [Test Suite & Quality Verification](#-test-suite--quality-verification)

---

## 🚀 Executive Summary

Customer churn represents one of the largest drivers of revenue loss in the telecommunications industry. Acquiring a new subscriber is estimated to cost **5x to 7x more** than retaining an existing customer.

This repository implements a production machine learning system that:
1. **Detects high-risk churners early** using gradient-boosted decision trees (XGBoost) and ensemble baselines.
2. **Optimizes decision thresholds for financial impact** by minimizing total expected business costs (weighing lost Customer Lifetime Value against retention campaign costs).
3. **Provides explainable predictions** with SHAP feature attributions at inference time to empower retention teams with actionable insights.
4. **Enforces production rigor** with automated data validation, a single serialized `ColumnTransformer` artifact, multi-Python CI/CD testing, and multi-stage container deployment.

---

## 🏗️ System Architecture

For a detailed architectural blueprint, refer to [architecture.md](architecture.md).

```mermaid
flowchart TD
    subgraph Data ["1. Ingestion & Validation"]
        A[Raw CSV Ingestion] --> B[Data Validation Suite<br/>23 Invariant Checks]
        B --> C[Stratified Train/Val/Test Split<br/>80/20 with 10% Val]
    end

    subgraph Preprocessing ["2. Deterministic Pipeline"]
        C --> D[fit_transform on Train only<br/>Ordinal + OHE + Imputation]
        D --> E[(Serialized pipeline.pkl)]
    end

    subgraph Modeling ["3. Modeling & Optimization"]
        D --> F1[Logistic Regression<br/>Linear Baseline]
        D --> F2[Random Forest<br/>Ensemble Baseline]
        D --> F3[XGBoost Classifier<br/>Primary Model]
        F3 --> G[Optuna Bayesian HPO<br/>Nested MLflow Runs]
        G --> H[(MLflow Experiment Tracking & Registry)]
    end

    subgraph DecisionLayer ["4. Evaluation & Business Logic"]
        H --> I[Multi-Metric Evaluation<br/>PR-AUC, ROC-AUC, Brier Score]
        I --> J[Financial Cost Matrix<br/>FN: $1,200 | FP: $50]
        J --> K[Cost-Optimal Threshold = 0.119]
    end

    subgraph Serving ["5. Serving & Explainability"]
        E -. Loaded at Startup .-> L[FastAPI Microservice]
        H -. Loaded at Startup .-> L
        L --> M[POST /predict]
        L --> N[POST /explain<br/>SHAP TreeExplainer]
        L --> O[Gradio UI /ui]
        L --> P[GET /health & /ready]
    end
```

---

## 💡 Key Engineering Highlights

| Capability | Implementation | Technical Advantage |
| :--- | :--- | :--- |
| **Train/Serve Parity** | `scikit-learn.compose.ColumnTransformer` | Single source of truth. Structurally eliminates feature skew between training and real-time inference without reliance on ad-hoc `pd.get_dummies`. |
| **Data Quality Gate** | Automated validation suite (`src/utils/validate_data.py`) | 23 rigorous checks covering missing values, unexpected category sets, tenure bounds, and cross-column business consistency. |
| **Hyperparameter Tuning** | `Optuna` + `MLflow` | Bayesian parameter optimization optimizing validation PR-AUC with automatic child run logging. |
| **Business Cost Layer** | Empirical validation sweep ($0.01 - 0.99, \Delta=0.001$) | Replaces the arbitrary 0.5 default threshold with an asymmetric cost optimization ($1,200 FN vs $50 FP). |
| **Explainable AI** | `SHAP TreeExplainer` | Deconstructs individual customer predictions into top positive and negative risk factors in sub-millisecond response times. |
| **Production API** | `FastAPI` (Pydantic v2) + `Gradio` | Schema-validated REST endpoints (`/predict`, `/explain`, `/health`, `/ready`) with mounted web UI (`/ui`). |
| **Containerization** | Multi-stage `dockerfile` | Security-hardened non-root user (`appuser`), Docker healthcheck, and minimal runtime footprint. |
| **CI/CD Automation** | `GitHub Actions` | Multi-Python testing matrix (3.11, 3.12, 3.13), Ruff linting, and automated Docker build test validation. |

---

## 📊 Model Exploration & Benchmarking

All models are trained using stratified splits and evaluated on an independent test set. Because telecom churn exhibits class imbalance (~26.5% positive rate), **PR-AUC (Precision-Recall Area Under Curve)** and **F1-Score** serve as primary decision metrics.

| Model Family | Pipeline Preprocessing | Test PR-AUC | Test ROC-AUC | Test F1-Score | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | StandardScaler + OHE | 0.635 | 0.842 | 0.600 | 0.138 |
| **Random Forest** (Baseline) | Tree (Unscaled) + OHE | 0.655 | 0.844 | 0.597 | 0.134 |
| **XGBoost Classifier** (Tuned) | Tree (Unscaled) + OHE | **0.655** | **0.841** | **0.634** | **0.135** |

---

## 💰 Business Decision Layer & Cost Optimization

Traditional classifiers use an uncalibrated default threshold of $0.50$. In real-world customer churn, misclassifications carry asymmetric economic costs:
- **False Negative (FN):** A churning customer is missed. **Lost Customer Lifetime Value (CLV) = $1,200**.
- **False Positive (FP):** A loyal customer is flagged. **Cost of unnecessary retention outreach / discounts = $50**.

```
Total Expected Cost = (FN × $1,200) + (FP × $50)
```

By sweeping thresholds between $0.01$ and $0.99$ on the validation set, the optimal operating threshold is identified at **$0.119$**:

| Operating Threshold | Precision | Recall | F1-Score | False Positives | False Negatives | Expected Cost / Customer |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.050 | 0.3667 | 0.9933 | 0.5356 | 512 | 2 | $37.33 |
| **0.119 (Optimal)** | **0.4727** | **0.9599** | **0.6337** | **321** | **12** | **$34.42** |
| 0.200 | 0.5332 | 0.9130 | 0.6732 | 239 | 26 | $37.52 |
| 0.350 | 0.6358 | 0.7760 | 0.6990 | 133 | 67 | $61.77 |
| 0.500 (Default) | 0.7077 | 0.5318 | 0.6074 | 66 | 140 | $121.57 |

> **Financial ROI**: Deploying the cost-optimal threshold of **0.119** achieves a **95.99% recall rate** on churners, reducing total churn-related financial losses by **71.7%** compared to the naive 0.50 cutoff ($34.42 vs. $121.57 per customer).

---

## 🔍 Explainable AI (SHAP)

Every churn prediction can be explained locally via the `/explain` endpoint:

```json
{
  "prediction": "Likely to churn",
  "churn_probability": 0.8531,
  "base_value": -0.0313,
  "top_contributing_factors": [
    {
      "feature": "tenure",
      "shap_value": 1.6971,
      "impact": "increases_churn_risk"
    },
    {
      "feature": "MonthlyCharges",
      "shap_value": 0.0930,
      "impact": "increases_churn_risk"
    },
    {
      "feature": "Contract_Month-to-month",
      "shap_value": 0.0812,
      "impact": "increases_churn_risk"
    },
    {
      "feature": "TotalCharges",
      "shap_value": -0.0450,
      "impact": "reduces_churn_risk"
    }
  ]
}
```

---

## 🔌 API Endpoints Reference

| Method | Path | Description | Sample Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root service descriptor and endpoint directory | `200 OK` |
| `GET` | `/health` | Service and model readiness probe | `200 OK` |
| `GET` | `/ready` | Container orchestration readiness probe | `200 OK` |
| `POST` | `/predict` | Predict churn outcome and probability | `200 OK` |
| `POST` | `/explain` | Predict and return top SHAP risk factors | `200 OK` |
| `GET` | `/ui` | Interactive Gradio prediction web interface | `200 OK` |
| `GET` | `/docs` | Interactive OpenAPI / Swagger documentation | `200 OK` |

---

## 🛠️ Getting Started & Local Reproduction

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/Telco-Customer-Churn-ML.git
cd Telco-Customer-Churn-ML

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Run Training Pipeline & Optuna HPO
```bash
# Execute pipeline with automatic business-cost threshold optimization
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Run with Optuna Bayesian hyperparameter optimization (e.g. 10 trials)
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --tune --n_trials 10
```

### 3. Launch MLflow Experiment Tracking UI
```bash
# In Windows PowerShell:
$env:MLFLOW_ALLOW_FILE_STORE="true"; python -m mlflow ui

# In Linux/macOS:
MLFLOW_ALLOW_FILE_STORE=true python -m mlflow ui

# Open http://localhost:5000 in your browser
```

### 4. Launch FastAPI Serving & Gradio UI
```bash
python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# API Documentation: http://localhost:8000/docs
# Interactive Gradio UI: http://localhost:8000/ui
```

---

## 🐳 Docker Deployment

Build and run the multi-stage, security-hardened container:

```bash
# Build image
docker build -t telco-churn:latest .

# Run container
docker run -d -p 8000:8000 --name telco-service telco-churn:latest

# Check health probe
curl http://localhost:8000/health
```

---

## 🧪 Test Suite & Quality Verification

The test suite contains **36 unit and integration tests** verifying data validation, preprocessing determinism, unknown category handling, model training, Optuna tuning, SHAP explainability, and FastAPI REST endpoints.

```


## 📜 License
This project is open-source and available under the [MIT License](LICENSE).