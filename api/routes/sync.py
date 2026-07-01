"""Sync route (POST /sync) — triggers a fresh ingestion run in the background.

Enqueues a single APScheduler job (the pipeline: list -> diff -> download -> parse
-> chunk -> embed -> upsert) and returns 202 with a job id immediately, so the
request doesn't block on a long sync. Overlapping triggers get 409.

GET /sync/status reports the live/last job state.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

from fastapi import APIRouter
from fastapi import status as http_status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from db import crud
from db.session import get_session
from ingestion.scheduler import get_state, set_auto_sync, trigger_sync

router = APIRouter(tags=["sync"])

# Poll cadence + safety cap for the progress stream (cap ≈ 30 min of a stuck job).
_STREAM_POLL_SECONDS = 0.4
_STREAM_MAX_TICKS = 4500


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data)}


class SyncRequest(BaseModel):
    folder_id: Optional[str] = None


class AutoSyncRequest(BaseModel):
    enabled: bool


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
    if result.get("status") == "error":
        return JSONResponse(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, content=result)
    return JSONResponse(status_code=http_status.HTTP_202_ACCEPTED, content=result)


@router.get("/sync/status")
def sync_status() -> dict:
    """Live state of the current/last sync job."""
    return get_state()


@router.post("/sync/auto")
def toggle_auto_sync(body: AutoSyncRequest) -> dict:
    """Enable/disable the background auto-sync job (persisted). Returns new state."""
    return set_auto_sync(body.enabled)


@router.get("/sync/stream")
async def sync_stream():
    """Stream live ingestion progress as SSE: `progress`* → `done`.

    Polls the in-memory scheduler state and emits a `progress` event
    ({running, done, total, file, phase}) each tick until the run finishes, then a
    terminal `done` event with the summary. The UI drives an `st.progress` bar from
    this. Safe to open with no sync running — it emits one snapshot then `done`.
    """

    async def gen():
        for _ in range(_STREAM_MAX_TICKS):
            state = get_state()
            prog = state.get("progress") or {}
            running = bool(state.get("running"))
            yield _sse(
                "progress",
                {
                    "running": running,
                    "done": prog.get("done", 0),
                    "total": prog.get("total", 0),
                    "file": prog.get("file"),
                    "phase": prog.get("phase"),
                },
            )
            if not running:
                break
            await asyncio.sleep(_STREAM_POLL_SECONDS)

        # `running` is True only if we fell out of the loop on the safety cap while
        # the job was still going — the UI must NOT report that as completed.
        final = get_state()
        yield _sse(
            "done",
            {
                "running": bool(final.get("running")),
                "summary": final.get("last_summary") or {},
                "finished_at": final.get("finished_at"),
                "last_error": final.get("last_error"),
            },
        )

    return EventSourceResponse(gen())
