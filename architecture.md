# 🏛️ Telco Customer Churn System Architecture

An in-depth architectural and engineering blueprint of the **Production Telecom Customer Churn ML Platform**, detailing system components, data lifecycle, zero-skew preprocessing invariants, Bayesian hyperparameter optimization, business-cost threshold optimization, SHAP explainability, and containerized serving.

---

## 📑 Architecture Overview

```mermaid
flowchart TD
    subgraph DataLayer ["1. Data Ingestion & Validation"]
        A[Raw Dataset: Telco-Customer-Churn.csv] --> B[Data Validation Suite<br/>23 Invariant Checks]
        B --> C[Stratified 80/20 Split<br/>Train: 4,507 | Val: 1,127 | Test: 1,409]
    end

    subgraph PreprocessingLayer ["2. Deterministic Preprocessing Pipeline"]
        C --> D[fit_transform on Train only<br/>Ordinal + OHE + Imputation]
        D --> E[(artifacts/pipeline.pkl<br/>Single Source of Truth)]
    end

    subgraph TrainingLayer ["3. Modeling & Optimization"]
        D --> F1[Logistic Regression<br/>Linear Baseline]
        D --> F2[Random Forest<br/>Ensemble Baseline]
        D --> F3[XGBoost Classifier<br/>Primary Model]
        F3 --> G[Optuna Bayesian HPO<br/>10-30 Trials on Validation PR-AUC]
        G --> H[(MLflow Tracking & Artifact Registry)]
        H --> I[(artifacts/model/model.pkl)]
    end

    subgraph BusinessLayer ["4. Financial Cost Optimization"]
        I --> J[Validation Cost Sweep<br/>Range: 0.01 - 0.99, Step: 0.001]
        J --> K[Cost Matrix: $1,200 FN vs $50 FP]
        K --> L[Optimal Business Threshold: 0.119<br/>Minimizes Expected Cost per Customer]
    end

    subgraph ServingLayer ["5. Serving & Inference Engine"]
        E -. Loaded at Startup .-> M[FastAPI Application]
        I -. Loaded at Startup .-> M
        M --> N1[GET /health & /ready<br/>Readiness & Liveness Probes]
        M --> N2[POST /predict<br/>Real-Time Churn Inference]
        M --> N3[POST /explain<br/>SHAP TreeExplainer Local Attribution]
        M --> N4[Gradio Web UI /ui<br/>Interactive Portal]
    end

    subgraph DeploymentLayer ["6. Packaging & CI/CD"]
        M --> O[Multi-Stage Dockerfile<br/>python:3.11-slim + Non-Root appuser]
        O --> P[GitHub Actions CI<br/>Linting, 36 Tests, Docker Build]
    end
```

---

## 1. Data Pipeline & Zero-Skew Preprocessing

### 1.1 Automated Validation Suite (`src/utils/validate_data.py`)
Before any data ingestion or modeling, the dataset passes through a strict 23-point validation gate:
- **Schema & Type Integrity**: Validates all 20 feature columns and target presence.
- **Range & Value Invariants**: `tenure >= 0`, `MonthlyCharges >= 0`, `TotalCharges >= 0`.
- **Categorical Domain Constraints**: Validates exact allowed categories for `gender`, `Contract`, `InternetService`, and all add-on services.
- **Business Logic Checks**: Confirms that when `PhoneService == 'No'`, `MultipleLines == 'No phone service'`; when `InternetService == 'No'`, all 6 security and streaming services equal `'No internet service'`.

### 1.2 Leakage-Free Preprocessing (`src/pipeline/preprocess.py`)
To eliminate train-serve feature skew, feature transformations are encapsulated in a single `sklearn.compose.ColumnTransformer`:
- **Binary Features (4)**: Encoded via `OrdinalEncoder` (`gender`, `Partner`, `Dependents`, `PaperlessBilling`).
- **Pass-through Numeric/Binary (1)**: `SeniorCitizen`.
- **Multi-Class Categoricals (11)**: Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`.
- **Continuous Numerics (3)**: Imputed via `SimpleImputer(strategy='median')` (`tenure`, `MonthlyCharges`, `TotalCharges`).
- **StandardScaler Integration**: Scaled strictly for linear models (Logistic Regression); unscaled for tree-based models (Random Forest, XGBoost).

```
Training Data ──► pipeline.fit_transform(X_train) ──► Saved as artifacts/pipeline.pkl
Validation Data ──► pipeline.transform(X_val)
Test Data ────────► pipeline.transform(X_test)
Live API Payload ──► pipeline.transform(X_payload)  (100% Deterministic Feature Order)
```

---

## 2. Hyperparameter Optimization & Experiment Tracking

### 2.1 Bayesian Optimization (`src/pipeline/tune.py`)
Optuna executes Tree-structured Parzen Estimator (TPE) sampling:
- **Objective**: Maximize Validation **PR-AUC (Precision-Recall Area Under Curve)**.
- **Hyperparameter Search Space**:
  - `max_depth`: `[3, 10]`
  - `learning_rate`: `[0.01, 0.3]` (log scale)
  - `n_estimators`: `[50, 300]`
  - `subsample`: `[0.6, 1.0]`
  - `colsample_bytree`: `[0.6, 1.0]`
  - `scale_pos_weight`: `[1.0, 5.0]` (tunes class weighting)

### 2.2 MLflow Tracking Hierarchy (`src/pipeline/train.py`)
- **Parent Run**: `XGBoost_Primary` logs dataset tags, validation PR-AUC, test metrics, serialized model, and business parameters.
- **Child Runs**: Each Optuna trial is logged as a nested child run recording trial parameters and intermediate validation metrics.

---

## 3. Financial Cost Matrix & Threshold Optimization

### 3.1 Business Cost Formulation (`src/pipeline/evaluate.py`)
Standard classifiers assume symmetric error penalties (default threshold $0.50$). In telecom churn, misclassification costs are highly asymmetric:
- **False Negative ($C_{\text{FN}} = \$1,200$)**: A churner is missed. Loss of remaining Customer Lifetime Value (CLV).
- **False Positive ($C_{\text{FP}} = \$50$)**: A loyal subscriber is targeted. Unnecessary retention discount / campaign spend.
- **True Positive / True Negative**: $C_{\text{TP}} = \$0$, $C_{\text{TN}} = \$0$.

$$\text{Total Cost}(\theta) = C_{\text{FN}} \cdot \text{FN}(\theta) + C_{\text{FP}} \cdot \text{FP}(\theta)$$
$$\text{Expected Cost per Customer}(\theta) = \frac{\text{Total Cost}(\theta)}{N}$$

### 3.2 Fine-Grained Validation Sweep
- **Search Range**: $\theta \in [0.01, 0.99]$ with step size $\Delta = 0.001$.
- **Validation Optimization**: The threshold $\theta^* = 0.119$ achieves the global cost minimum on the validation set and is frozen before test set evaluation.

| Threshold ($\theta$) | Precision | Recall | F1-Score | FP | FN | Expected Cost / Customer |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.050 | 0.3667 | 0.9933 | 0.5356 | 512 | 2 | $37.33 |
| **0.119 (Optimal)** | **0.4727** | **0.9599** | **0.6337** | **321** | **12** | **$34.42** |
| 0.200 | 0.5332 | 0.9130 | 0.6732 | 239 | 26 | $37.52 |
| 0.350 | 0.6358 | 0.7760 | 0.6990 | 133 | 67 | $61.77 |
| 0.500 (Default) | 0.7077 | 0.5318 | 0.6074 | 66 | 140 | $121.57 |

> **Financial Impact**: Optimizing the decision threshold to **0.119** yields a **71.7% cost reduction** compared to the naive 0.50 default threshold ($34.42 vs. $121.57 per customer).

---

## 4. Real-Time Explainability Engine (SHAP)

### 4.1 Local Attribution (`src/pipeline/explain.py`)
The system initializes a `shap.TreeExplainer` over the trained XGBoost booster:
- Computes exact Shapley values for incoming customer feature vectors.
- Calculates the expected base probability (population prior).
- Ranks positive drivers (increasing churn probability) and protective factors (reducing churn probability).

```
Customer Vector ──► Preprocessing Transform ──► TreeExplainer.shap_values() ──► JSON Attribution
```

---

## 5. Production Serving & Runtime Architecture

### 5.1 FastAPI Microservice Layer (`src/app/main.py`)
- **Pydantic v2 Validation (`src/api/schemas.py`)**: Validates input types, constraints, and defaults.
- **REST Endpoints**:
  - `GET /health` & `GET /ready`: Container health and readiness monitoring.
  - `POST /predict`: Real-time probability scoring and thresholded classification.
  - `POST /explain`: Churn prediction with top 5 feature SHAP attributions.
- **Gradio Interactive UI**: Mounted directly at `/ui` for business stakeholder interaction.

### 5.2 Container Isolation (`dockerfile`)
- **Stage 1 (Builder)**: Installs dependencies using wheels into `/root/.local`.
- **Stage 2 (Runner)**: Minimal `python:3.11-slim` runtime copying prebuilt wheels into non-root user directory `/home/appuser/.local`.
- **Security Invariant**: Runs under non-root UID `1001` (`appuser`).
- **Healthcheck**: Configured to poll `http://localhost:8000/health` every 30s.

---

## 6. Continuous Integration & Quality Assurance

### 6.1 GitHub Actions Workflow (`.github/workflows/ci.yml`)
1. **Linting Job**: Ruff code formatting and lint checks.
2. **Multi-Python Test Matrix**: Executes all 36 unit and integration tests across **Python 3.11, 3.12, and 3.13**.
3. **Docker Build Test**: Automatically builds the container image to verify zero build regressions.

```
git push ──► CI Matrix (3.11, 3.12, 3.13) ──► Pytest (36 Tests) ──► Docker Build Test ──► Green Build
```
