"""APScheduler-based background ingestion (in-process; no broker, no extra container).

A single BackgroundScheduler runs:
  - manual sync jobs triggered via POST /sync (run immediately), and
  - an auto-sync job on a fixed interval (settings.auto_sync_minutes, default 15).

An in-memory state guard rejects overlapping runs (manual or auto) so concurrent
syncs can't corrupt state. Auto-sync is skipped while Drive isn't configured yet,
and records its completion time as `last_auto_sync` for the UI.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from config import settings
from db import crud
from db.session import get_session
from ingestion.pipeline import run_sync

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()
_state: dict = {
    "running": False,
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "last_summary": None,
    "last_error": None,
    "last_auto_sync": None,  # ISO time the last AUTO-sync finished
    "last_auto_summary": None,  # summary dict of the last auto-sync
    "auto_sync_minutes": settings.auto_sync_minutes,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_scheduler() -> None:
    """Start the scheduler + the recurring auto-sync job (from the app lifespan)."""
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.start()
    _scheduler.add_job(
        _auto_sync_job,
        "interval",
        minutes=settings.auto_sync_minutes,
        id="auto_sync",
        max_instances=1,
        coalesce=True,
    )
    logger.info("APScheduler started; auto-sync every %s min.", settings.auto_sync_minutes)


def shutdown_scheduler() -> None:
    """Stop the scheduler on app shutdown."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_state() -> dict:
    """Snapshot of the current/last sync job state (incl. last auto-sync time)."""
    with _lock:
        return dict(_state)


def _config_ready() -> bool:
    """True if a Drive folder + service-account credentials are configured (cheap,
    no network) — keeps auto-sync quiet until the user has set things up."""
    session = get_session()
    try:
        cfg = crud.get_or_create_config(session)
        folder = cfg.drive_folder_id or settings.drive_folder_id
        has_creds = bool(crud.get_service_account_info(cfg)) or os.path.exists(
            settings.google_service_account_path
        )
        return bool(folder) and bool(has_creds)
    finally:
        session.close()


def trigger_sync(folder_id: str | None = None) -> dict:
    """Enqueue a single manual sync job; reject if one is already running."""
    with _lock:
        if _state["running"]:
            return {"job_id": _state["job_id"], "status": "running", "already_running": True}
        job_id = f"sync_{uuid.uuid4().hex[:8]}"
        _state.update(
            running=True, job_id=job_id, started_at=_now_iso(), finished_at=None, last_error=None
        )

    if _scheduler is None:
        start_scheduler()
    try:
        _scheduler.add_job(_run_job, args=[folder_id, job_id], id=job_id, max_instances=1)
    except Exception as exc:  # never leave `running` stuck if enqueue fails
        logger.exception("Failed to enqueue sync job")
        with _lock:
            _state["running"] = False
            _state["last_error"] = str(exc)
        return {"status": "error", "error": "Could not start the sync job."}
    return {"job_id": job_id, "status": "running"}


def _run_job(folder_id: str | None, job_id: str) -> None:
    try:
        summary = run_sync(folder_id)
        with _lock:
            _state["last_summary"] = summary
            _state["last_error"] = (
                summary.get("error") if summary.get("status") == "error" else None
            )
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Sync job %s crashed", job_id)
        with _lock:
            _state["last_error"] = str(exc)
            _state["last_summary"] = {"status": "error", "error": str(exc)}
    finally:
        with _lock:
            _state["running"] = False
            _state["finished_at"] = _now_iso()


def _auto_sync_job() -> None:
    """Interval-triggered auto-sync. Skips when Drive isn't configured or a sync is
    already running; records the completion time as last_auto_sync for the UI."""
    if not _config_ready():
        return
    with _lock:
        if _state["running"]:
            return  # don't overlap a manual/auto run already in progress
        job_id = f"autosync_{uuid.uuid4().hex[:8]}"
        _state.update(
            running=True, job_id=job_id, started_at=_now_iso(), finished_at=None, last_error=None
        )

    logger.info("Auto-sync %s starting.", job_id)
    try:
        summary = run_sync(None)
        with _lock:
            _state["last_summary"] = summary
            _state["last_auto_summary"] = summary
            _state["last_error"] = (
                summary.get("error") if summary.get("status") == "error" else None
            )
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception("Auto-sync job %s crashed", job_id)
        with _lock:
            _state["last_error"] = str(exc)
            _state["last_summary"] = {"status": "error", "error": str(exc)}
    finally:
        with _lock:
            _state["last_auto_sync"] = _now_iso()
            _state["running"] = False
            _state["finished_at"] = _now_iso()
