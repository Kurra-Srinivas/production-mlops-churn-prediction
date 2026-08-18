# 📊 Telco Customer Churn Prediction & MLOps

An end-to-end Machine Learning project to predict telecom customer churn using **XGBoost**, **Optuna** Bayesian optimization, **MLflow** experiment tracking, **business-cost threshold optimization**, **SHAP** explainability, and **FastAPI / Gradio** serving.

---

## 🚀 Features

- **Leakage-Free Preprocessing**: Unified `ColumnTransformer` (Ordinal, One-Hot Encoding, and SimpleImputer) reused across training, validation, and inference.
- **Model Tuning**: Bayesian hyperparameter search using Optuna optimizing validation PR-AUC.
- **Business Cost Optimization**: Selects the cost-optimal decision threshold on validation data ($1,200 False Negative vs. $50 False Positive cost matrix).
- **Explainability**: Real-time SHAP feature attributions on predictions.
- **FastAPI & Gradio**: REST API endpoints (`/health`, `/ready`, `/predict`, `/explain`) with an interactive Gradio UI mounted at `/ui`.
- **Containerized**: Production multi-stage Docker build running under non-root user.
- **Automated Testing**: 36 unit and integration tests passing.

---

## 📈 Model Performance & Business Metrics

Evaluated on an untouched holdout test set (1,409 customers):

| Metric | XGBoost (Tuned) | Baseline (Logistic Regression) |
| :--- | :---: | :---: |
| **PR-AUC** | **0.655** | 0.635 |
| **ROC-AUC** | **0.841** | 0.842 |
| **Recall (Churners)** | **95.99%** | 78.82% |
| **Optimal Threshold** | **0.119** | 0.350 |
| **Expected Cost / Customer** | **$34.42** | $60.10 |

---

## 🛠️ Quick Start

### 1. Installation

```bash
git clone https://github.com/Kurra-Srinivas/production-mlops-churn-prediction.git
cd production-mlops-churn-prediction

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Run Training Pipeline

```bash
# Run pipeline with automatic business-cost threshold optimization
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Run with Optuna hyperparameter tuning (10 trials)
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --tune --n_trials 10
```

### 3. Launch MLflow Experiment UI

```bash
# Windows PowerShell:
$env:MLFLOW_ALLOW_FILE_STORE="true"; python -m mlflow ui

# Linux/macOS:
MLFLOW_ALLOW_FILE_STORE=true python -m mlflow ui
```
Open **http://localhost:5000** to view runs, parameters, and metrics.

### 4. Run FastAPI Application & Web UI

```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web UI**: [http://localhost:8000/ui](http://localhost:8000/ui)
- **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Run with Docker

```bash
# Build image
docker build -t telco-churn:latest .

# Run container
docker run -d -p 8000:8000 --name telco-service telco-churn:latest
```

Open [http://localhost:8000/ui](http://localhost:8000/ui) in your browser.

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
├── artifacts/              # Serialized pipeline.pkl and model.pkl
├── configs/                # Business and model YAML configurations
├── data/
│   └── raw/                # Telco-Customer-Churn.csv dataset
├── dockerfile              # Multi-stage Docker container definition
├── notebooks/              # Exploratory Data Analysis (EDA.ipynb)
├── scripts/
│   ├── download_data.py    # Dataset fetching script
│   └── run_pipeline.py     # Training and tuning CLI entrypoint
├── src/
│   ├── api/                # FastAPI routers and Pydantic schemas
│   ├── app/                # Application entrypoint & Gradio UI mount
│   ├── pipeline/           # Preprocessing, train, tune, evaluate, explain
│   ├── serving/            # Inference engine
│   └── utils/              # Data validation and helper functions
├── tests/                  # 36 Unit and integration tests
├── architecture.md         # Detailed architectural blueprint
├── requirements.txt        # Production dependencies
└── requirements-dev.txt    # Testing & development dependencies
```

---

## 📜 License
MIT License.