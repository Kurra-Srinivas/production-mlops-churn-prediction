"""
MLFLOW MODEL REGISTRY
=====================

Handles version control, model registration, lifecycle stage transitions (Staging -> Production),
and production artifact export.
"""

import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient


def register_model_version(
    run_id: str,
    model_name: str = "ChurnPredictor",
    tags: dict = None,
) -> str:
    """
    Register a logged model artifact from an MLflow run to the Model Registry.
    """
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    if tags:
        for k, v in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=k,
                value=str(v),
            )

    # Set initial stage to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"✅ Model {model_name} version {model_version.version} registered & moved to Staging")
    return str(model_version.version)


def promote_model_to_production(
    model_name: str = "ChurnPredictor",
    version: str = "1",
) -> None:
    """
    Promote a specific model version to Production and archive previous production versions.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"🚀 Model {model_name} version {version} promoted to Production")


def get_production_model_uri(model_name: str = "ChurnPredictor") -> str:
    """
    Return the MLflow URI for the current production version of the model.
    """
    return f"models:/{model_name}/Production"
