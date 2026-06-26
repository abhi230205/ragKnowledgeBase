"""Status route (GET /status) — knowledge-base stats for the UI dashboard.

Reports document count, total chunks, last sync time, per-file errors, and the
live sync-job state. Chunk count comes from Chroma (authoritative) with the DB
sum as a fallback. Stays off the embedding model (cheap-ish; no model load).
"""

from __future__ import annotations

import logging

from db import crud
from db.session import get_session
from fastapi import APIRouter
from ingestion.scheduler import get_state
from vectorstore import chroma_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["status"])


@router.get("/status")
def get_status() -> dict:
    """Knowledge-base status + live sync state."""
    session = get_session()
    try:
        summary = crud.status_summary(session)
    finally:
        session.close()

    # Prefer Chroma's authoritative chunk count; fall back to the DB sum.
    try:
        summary["chunks"] = chroma_store.count(chroma_store.get_collection())
    except Exception as exc:  # pragma: no cover - vector store optional at boot
        logger.warning("Could not read Chroma count: %s", exc)

    summary["sync"] = get_state()
    return summary
