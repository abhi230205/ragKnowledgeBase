"""Health-check route — cheap, dependency-free, always 200 when the app is up.

Used by the Docker healthcheck (compose) and by the Streamlit UI to confirm the
API is reachable before issuing requests.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
