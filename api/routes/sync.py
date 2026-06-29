"""Sync route (POST /sync) — triggers a fresh ingestion run in the background.

Enqueues a single APScheduler job (the pipeline: list -> diff -> download -> parse
-> chunk -> embed -> upsert) and returns 202 with a job id immediately, so the
request doesn't block on a long sync. Overlapping triggers get 409.

GET /sync/status reports the live/last job state.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from db import crud
from db.session import get_session
from ingestion.scheduler import get_state, trigger_sync

router = APIRouter(tags=["sync"])


class SyncRequest(BaseModel):
    folder_id: Optional[str] = None


@router.post("/sync")
def post_sync(body: Optional[SyncRequest] = None):
    """Start a background ingestion run (202). 409 if one is already running.

    Cheap pre-flight (no network): a missing Drive folder id or service-account
    credentials returns a clear 422 instead of a 202 that silently does nothing.
    """
    folder_id = body.folder_id if body else None
    session = get_session()
    try:
        cfg = crud.get_or_create_config(session)
        folder = folder_id or cfg.drive_folder_id or settings.drive_folder_id
        has_creds = bool(crud.get_service_account_info(cfg)) or os.path.exists(
            settings.google_service_account_path
        )
    finally:
        session.close()

    if not folder:
        return JSONResponse(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "error": "Set a Drive folder ID in Settings before syncing.",
            },
        )
    if not has_creds:
        return JSONResponse(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "error",
                "error": "Upload the Google service-account JSON in Settings before syncing.",
            },
        )

    result = trigger_sync(folder_id)
    if result.get("already_running"):
        return JSONResponse(status_code=http_status.HTTP_409_CONFLICT, content=result)
    return JSONResponse(status_code=http_status.HTTP_202_ACCEPTED, content=result)


@router.get("/sync/status")
def sync_status() -> dict:
    """Live state of the current/last sync job."""
    return get_state()
