"""APScheduler-based background ingestion (in-process; no broker, no extra container).

A single BackgroundScheduler runs the sync job off the request path. An in-memory
state guard rejects overlapping runs so concurrent /sync triggers can't corrupt
state (the brief's concurrent-sync edge case). The job runs once immediately when
triggered (add_job with no trigger = run now).
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

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
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_scheduler() -> None:
    """Start the background scheduler (called from the app lifespan)."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        _scheduler.start()
        logger.info("APScheduler started.")


def shutdown_scheduler() -> None:
    """Stop the scheduler on app shutdown."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def get_state() -> dict:
    """Snapshot of the current/last sync job state."""
    with _lock:
        return dict(_state)


def trigger_sync(folder_id: str | None = None) -> dict:
    """Enqueue a single background sync job; reject if one is already running."""
    with _lock:
        if _state["running"]:
            return {
                "job_id": _state["job_id"],
                "status": "running",
                "already_running": True,
            }
        job_id = f"sync_{uuid.uuid4().hex[:8]}"
        _state.update(
            running=True,
            job_id=job_id,
            started_at=_now_iso(),
            finished_at=None,
            last_error=None,
        )

    if _scheduler is None:
        start_scheduler()
    _scheduler.add_job(_run_job, args=[folder_id, job_id], id=job_id, max_instances=1)
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
