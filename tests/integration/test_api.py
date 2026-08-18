"""
Integration tests for FastAPI REST endpoints (/health, /predict, /explain).
"""

import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "/predict" in data["endpoints"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert data["ready"] is True


def test_predict_endpoint_valid_payload(sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Likely to churn", "Not likely to churn"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "threshold_used" in data


def test_predict_endpoint_invalid_payload():
    # Missing required fields
    response = client.post("/predict", json={"gender": "Male"})
    assert response.status_code == 422  # Pydantic schema validation failure


def test_explain_endpoint_valid_payload(sample_customer):
    response = client.post("/explain", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert "top_contributing_factors" in data
    assert len(data["top_contributing_factors"]) > 0
    assert "feature" in data["top_contributing_factors"][0]
    assert "shap_value" in data["top_contributing_factors"][0]
