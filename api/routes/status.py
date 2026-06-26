"""Status route (GET /status) — knowledge-base stats for the UI dashboard.

TODO (Phase 2): return real counts from SQLite + Chroma (document count, total
chunks, last sync time, recent per-file errors). Returns zeros until the
ingestion pipeline lands so the dashboard renders cleanly from Day 1.

Planned response:
    {"documents": 12, "chunks": 148, "last_sync": "2026-06-25T09:14:03Z",
     "errors": [{"file": "scan_only.pdf", "reason": "no_extractable_text"}]}
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["status"])


@router.get("/status")
def get_status() -> dict:
    """Knowledge-base status. TODO: read real counts (Phase 2)."""
    return {"documents": 0, "chunks": 0, "last_sync": None, "errors": []}
