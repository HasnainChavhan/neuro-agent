"""
NeuroAgent — Health Route
"""
from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.models.response import HealthResponse
from app.utils.metrics import metrics_store

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Infrastructure"],
)
async def health() -> HealthResponse:
    """Returns service health status and runtime metrics."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment,
        mock_mode=settings.mock_llm,
        metrics=metrics_store.summary(),
    )
