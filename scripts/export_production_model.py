#!/usr/bin/env python3
"""
EXPORT PRODUCTION MODEL ARTIFACTS
=================================

Utility script for CI/CD container builds.
Extracts the latest Production model artifacts + preprocessing pipeline from MLflow
and places them in a self-contained directory for Docker building.
"""

import os
import shutil
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def export_production_artifacts(
    model_name: str = "ChurnPredictor",
    output_dir: str = "model_artifacts",
    mlflow_uri: str = None,
):
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()
    os.makedirs(output_dir, exist_ok=True)

    # Find latest Production version
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        # Fallback to Staging or latest version if no Production stage
        latest_versions = client.get_latest_versions(model_name, stages=["Staging", "None"])
        if not latest_versions:
            print(f"⚠️  No registered versions found for '{model_name}'. Copying local artifacts...")
            if os.path.exists("artifacts"):
                shutil.copytree("artifacts", output_dir, dirs_exist_ok=True)
            return

    prod_version = latest_versions[0]
    run_id = prod_version.run_id
    print(f"📦 Exporting artifacts from Run ID: {run_id} (Version {prod_version.version})")

    # Download model artifact
    local_model_path = client.download_artifacts(run_id, "model", output_dir)
    print(f"✅ Model exported to: {local_model_path}")

    # Copy pipeline.pkl and feature_columns.txt if present
    for art in ["pipeline.pkl", "feature_columns.txt"]:
        try:
            client.download_artifacts(run_id, art, output_dir)
        except Exception:
            local_fallback = os.path.join("artifacts", art)
            if os.path.exists(local_fallback):
                shutil.copy2(local_fallback, os.path.join(output_dir, art))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export Production Model Artifacts")
    p.add_argument("--model_name", type=str, default="ChurnPredictor")
    p.add_argument("--output_dir", type=str, default="model_artifacts")
    p.add_argument("--mlflow_uri", type=str, default=None)
    args = p.parse_args()
    export_production_artifacts(args.model_name, args.output_dir, args.mlflow_uri)
