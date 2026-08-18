"""
HEALTH & READINESS ENDPOINTS
============================
"""

from fastapi import APIRouter

from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Service health check verifying model and pipeline readiness.
    """
    from src.serving.inference import model, pipeline

    model_ok = model is not None
    pipeline_ok = pipeline is not None

    return HealthResponse(
        status="ok" if (model_ok and pipeline_ok) else "degraded",
        model_loaded=model_ok,
        pipeline_loaded=pipeline_ok,
    )


@router.get("/ready")
def readiness_check():
    """
    Kubernetes / Container readiness probe endpoint.
    """
    from src.serving.inference import model, pipeline

    if model is None or pipeline is None:
        return {"ready": False}, 503
    return {"ready": True}
