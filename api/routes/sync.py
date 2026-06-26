"""Sync route (POST /sync) — triggers a fresh ingestion run in the background.

Enqueues a single APScheduler job (the pipeline: list -> diff -> download -> parse
-> chunk -> embed -> upsert) and returns 202 with a job id immediately, so the
request doesn't block on a long sync. Overlapping triggers get 409.

GET /sync/status reports the live/last job state.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from ingestion.scheduler import get_state, trigger_sync
from pydantic import BaseModel

router = APIRouter(tags=["sync"])


class SyncRequest(BaseModel):
    folder_id: Optional[str] = None


@router.post("/sync")
def post_sync(body: Optional[SyncRequest] = None):
    """Start a background ingestion run (202). 409 if one is already running."""
    folder_id = body.folder_id if body else None
    result = trigger_sync(folder_id)
    if result.get("already_running"):
        return JSONResponse(status_code=http_status.HTTP_409_CONFLICT, content=result)
    return JSONResponse(status_code=http_status.HTTP_202_ACCEPTED, content=result)


@router.get("/sync/status")
def sync_status() -> dict:
    """Live state of the current/last sync job."""
    return get_state()
