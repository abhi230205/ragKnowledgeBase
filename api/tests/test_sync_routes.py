"""Route-level tests for the sync endpoints that lacked coverage:
GET /sync/status (auto-sync fields), POST /sync/auto (toggle + persistence),
and the GET /sync/stream SSE progress channel (progress* -> done, incl. the
honest-terminal behaviour when the stream hits its safety cap mid-run).

Only Drive is faked; these exercise the real FastAPI routes + scheduler state.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as dbs
import ingestion.scheduler as sched
from db import crud
from db.models import Base
from main import app
from routes import sync as sync_route

client = TestClient(app)


def _baseline_state() -> dict:
    return {
        "running": False,
        "job_id": None,
        "started_at": None,
        "finished_at": None,
        "last_summary": None,
        "last_error": None,
        "last_auto_sync": None,
        "last_auto_summary": None,
        "auto_sync_minutes": 15,
        "auto_sync_enabled": True,
        "progress": None,
    }


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    """Fresh temp DB (so config writes work) + clean scheduler state per test."""
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'sync.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(dbs, "engine", engine)
    monkeypatch.setattr(dbs, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(sched, "_state", _baseline_state())
    yield


def _sse_events(text: str):
    out, event = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            try:
                out.append((event, json.loads(line.split(":", 1)[1].strip())))
            except json.JSONDecodeError:
                pass
    return out


# ---------------------------------------------------------------- /sync/status


def test_sync_status_exposes_auto_sync_fields():
    body = client.get("/sync/status").json()
    assert body["auto_sync_enabled"] is True
    assert body["auto_sync_minutes"] == 15
    assert "progress" in body and body["progress"] is None


# ---------------------------------------------------------------- /sync/auto


def test_auto_sync_route_toggles_and_persists():
    off = client.post("/sync/auto", json={"enabled": False})
    assert off.status_code == 200 and off.json()["auto_sync_enabled"] is False
    # persisted to the config row...
    s = dbs.SessionLocal()
    assert crud.get_or_create_config(s).auto_sync_enabled is False
    s.close()
    # ...and visible via /sync/status
    assert client.get("/sync/status").json()["auto_sync_enabled"] is False

    on = client.post("/sync/auto", json={"enabled": True})
    assert on.json()["auto_sync_enabled"] is True


def test_auto_sync_route_requires_enabled_field():
    assert client.post("/sync/auto", json={}).status_code == 422


# ---------------------------------------------------------------- /sync/stream


def test_sync_stream_idle_emits_progress_then_done():
    """With no sync running, the stream emits one progress snapshot then done."""
    r = client.get("/sync/stream")
    assert r.status_code == 200
    events = _sse_events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "progress" and kinds[-1] == "done"
    first = events[0][1]
    assert first["running"] is False and first["total"] == 0
    done = events[-1][1]
    assert done["running"] is False  # a genuine completion, not a cap


def test_sync_stream_reports_progress_until_done(monkeypatch):
    """A running sync streams progress ticks and ends with a running:false done."""
    seq = [
        {
            "running": True,
            "progress": {"done": 1, "total": 2, "file": "a.pdf", "phase": "embedding"},
        },
        {
            "running": True,
            "progress": {"done": 2, "total": 2, "file": "b.pdf", "phase": "embedding"},
        },
        {
            "running": False,
            "progress": {"done": 2, "total": 2, "file": "b.pdf", "phase": "embedding"},
        },
        {"running": False, "last_summary": {"summary": {"added": 2}}, "finished_at": "t"},
    ]
    calls = {"i": 0}

    def fake_get_state():
        i = min(calls["i"], len(seq) - 1)
        calls["i"] += 1
        return seq[i]

    monkeypatch.setattr(sync_route, "get_state", fake_get_state)
    monkeypatch.setattr(sync_route, "_STREAM_POLL_SECONDS", 0)

    events = _sse_events(client.get("/sync/stream").text)
    progs = [d for e, d in events if e == "progress"]
    assert any(p["running"] and p["done"] == 1 for p in progs)  # mid-run tick
    assert progs[-1]["running"] is False  # final poll saw completion
    done = events[-1]
    assert done[0] == "done" and done[1]["running"] is False
    assert done[1]["summary"] == {"summary": {"added": 2}}


def test_sync_stream_done_flags_still_running_when_capped(monkeypatch):
    """If the stream hits its tick cap while the job is still running, the terminal
    `done` must carry running:true so the UI does NOT report a false completion."""
    monkeypatch.setattr(sync_route, "get_state", lambda: {"running": True, "progress": None})
    monkeypatch.setattr(sync_route, "_STREAM_POLL_SECONDS", 0)
    monkeypatch.setattr(sync_route, "_STREAM_MAX_TICKS", 3)

    events = _sse_events(client.get("/sync/stream").text)
    assert [e for e, _ in events].count("progress") == 3  # capped at MAX_TICKS
    done = events[-1]
    assert done[0] == "done" and done[1]["running"] is True
