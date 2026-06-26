"""APScheduler-based background ingestion (in-process, no broker, no extra container).

TODO (Phase 2):
- A BackgroundScheduler with a single sync job configured max_instances=1 (and an
  in-DB guard) so concurrent /sync triggers can't run overlapping ingestions and
  corrupt state.
- Triggered by POST /sync (manual) and optionally on an interval.
- Runs the full pipeline off the request path:
      list -> diff -> (download -> parse -> chunk -> embed -> upsert) per file,
      with per-file try/except so one bad file doesn't fail the batch.
"""

from __future__ import annotations


def start_scheduler() -> None:
    """Start the background scheduler on app startup. TODO: implement (Phase 2)."""
    raise NotImplementedError("scheduler.start_scheduler — Phase 2")


def run_sync_job() -> dict:
    """Execute one ingestion run and return a summary. TODO: implement (Phase 2)."""
    raise NotImplementedError("scheduler.run_sync_job — Phase 2")
