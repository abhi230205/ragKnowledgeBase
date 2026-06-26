"""Sync route (POST /sync) — triggers a fresh ingestion run.

TODO (Phase 2): enqueue a single APScheduler job (no overlapping runs) that runs
the Drive -> vector-store pipeline:
    list (recursive) -> diff (added/modified/deleted/unchanged) -> download ->
    parse (PyMuPDF/pdfplumber) -> chunk (overlap, page-tracked) -> embed -> upsert.
Returns 202 with a job id so the HTTP request doesn't block on a long sync.

Planned response (202): {"job_id": "sync_8f12", "status": "running"}
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["sync"])


@router.post("/sync")
def trigger_sync():
    """Start a background ingestion run. TODO: implement (Phase 2)."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "POST /sync not implemented yet (Phase 2)"},
    )
