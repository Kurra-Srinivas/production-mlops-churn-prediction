"""
INFERENCE PIPELINE
==================

Loads the trained model and fitted sklearn preprocessing pipeline at startup,
then uses them to predict customer churn from raw input data.
"""

import os
import glob
import joblib
import pandas as pd
import mlflow

from src.pipeline.preprocess import load_pipeline

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

MODEL_DIR     = os.environ.get("MODEL_DIR",     "/app/model")
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "/app/model/pipeline.pkl")
THRESHOLD     = float(os.environ.get("PREDICTION_THRESHOLD", "0.35"))


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def _load_pipeline():
    """Load the fitted sklearn preprocessing pipeline."""
    if os.path.exists(PIPELINE_PATH):
        return load_pipeline(PIPELINE_PATH)

    local_pipeline_paths = glob.glob("./artifacts/pipeline.pkl")
    if local_pipeline_paths:
        path = local_pipeline_paths[0]
        print(f"[OK] Pipeline loaded from {path}")
        return load_pipeline(path)

    print("[WARN] sklearn pipeline artifact not found at startup.")
    return None


pipeline = _load_pipeline()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model():
    """Load the trained model (joblib or MLflow pyfunc)."""
    # Priority 1: Check MODEL_DIR/model.pkl
    pkl_path = os.path.join(MODEL_DIR, "model.pkl")
    if os.path.exists(pkl_path):
        try:
            m = joblib.load(pkl_path)
            print(f"[OK] Model loaded from {pkl_path}")
            return m
        except Exception:
            pass

    # Priority 2: Check ./artifacts/model/model.pkl
    if os.path.exists("./artifacts/model/model.pkl"):
        try:
            m = joblib.load("./artifacts/model/model.pkl")
            print("[OK] Model loaded from ./artifacts/model/model.pkl")
            return m
        except Exception:
            pass

    # Priority 3: Check MODEL_DIR as MLflow model
    if os.path.exists(MODEL_DIR):
        try:
            m = mlflow.pyfunc.load_model(MODEL_DIR)
            print(f"[OK] MLflow model loaded from {MODEL_DIR}")
            return m
        except Exception:
            pass

    # Priority 4: Search mlruns / models
    mlflow_models = glob.glob("./mlruns/*/*/artifacts/model") + glob.glob("./mlruns/*/models/*/artifacts/model")
    if mlflow_models:
        try:
            latest = max(mlflow_models, key=os.path.getmtime)
            m = mlflow.pyfunc.load_model(latest)
            print(f"[OK] MLflow model loaded from {latest}")
            return m
        except Exception:
            pass

    print("[WARN] Model artifact not found at startup.")
    return None


model = _load_model()


# ---------------------------------------------------------------------------
# Feature schema loading
# ---------------------------------------------------------------------------

def _load_feature_names() -> list[str]:
    """Load the feature column list written during training."""
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if os.path.exists(feature_file):
        with open(feature_file) as f:
            return [ln.strip() for ln in f if ln.strip()]

    local_ff = "./artifacts/feature_columns.txt"
    if os.path.exists(local_ff):
        with open(local_ff) as f:
            return [ln.strip() for ln in f if ln.strip()]

    return []


FEATURE_COLS = _load_feature_names()


# ---------------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------------

def predict(input_dict: dict) -> str:
    """
    Predict customer churn from raw customer data.
    """
    global model, pipeline
    if model is None or pipeline is None:
        raise RuntimeError(
            "Model or preprocessing pipeline is not initialized. "
            "Please ensure model artifacts exist."
        )

    df = pd.DataFrame([input_dict])

    try:
        X = pipeline.transform(df)
    except Exception as e:
        raise RuntimeError(f"Feature transformation failed: {e}") from e

    try:
        raw_model = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else model
        if hasattr(raw_model, "predict_proba"):
            prob = float(raw_model.predict_proba(X)[0, 1])
        else:
            preds = raw_model.predict(X)
            if hasattr(preds, "tolist"):
                preds = preds.tolist()
            prob = float(preds[0] if isinstance(preds, (list, tuple)) else preds)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    return "Likely to churn" if prob >= THRESHOLD else "Not likely to churn"
