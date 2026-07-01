"""Tests for the auto-sync scheduler logic (no real Drive; no APScheduler tick)."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as dbs
import ingestion.scheduler as sched
from config import settings
from db import crud
from db.models import Base


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Isolate the module-global sync state per test."""
    baseline = {
        "running": False,
        "job_id": None,
        "started_at": None,
        "finished_at": None,
        "last_summary": None,
        "last_error": None,
        "last_auto_sync": None,
        "last_auto_summary": None,
        "auto_sync_minutes": settings.auto_sync_minutes,
        "auto_sync_enabled": True,
        "progress": None,
    }
    monkeypatch.setattr(sched, "_state", baseline)
    yield


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(dbs, "engine", engine)
    monkeypatch.setattr(dbs, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(settings, "drive_folder_id", None)
    monkeypatch.setattr(settings, "google_service_account_path", str(tmp_path / "nope.json"))
    return engine


def test_state_exposes_last_auto_sync():
    assert "last_auto_sync" in sched.get_state()


def test_migrate_adds_auto_sync_column(tmp_path, monkeypatch):
    """_migrate() adds auto_sync_enabled to a config table created before the column."""
    eng = create_engine(
        f"sqlite:///{(tmp_path / 'old.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    with eng.begin() as conn:  # simulate an older schema (no auto_sync_enabled)
        conn.exec_driver_sql("CREATE TABLE config (id INTEGER PRIMARY KEY, top_k INTEGER)")
        conn.exec_driver_sql("INSERT INTO config (id, top_k) VALUES (1, 5)")
    monkeypatch.setattr(dbs, "engine", eng)

    dbs._migrate()

    with eng.begin() as conn:
        cols = [r[1] for r in conn.exec_driver_sql("PRAGMA table_info('config')").fetchall()]
    assert "auto_sync_enabled" in cols


def test_config_ready_false_when_unconfigured(temp_db):
    assert sched._config_ready() is False


def test_config_ready_true_when_configured(temp_db):
    s = dbs.SessionLocal()
    crud.update_config(
        s,
        drive_folder_id="FOLDER",
        service_account_json=json.dumps({"client_email": "x@y.iam", "private_key": "k"}),
    )
    s.close()
    assert sched._config_ready() is True


def test_auto_sync_skips_when_unconfigured(monkeypatch):
    calls = {"n": 0}

    def _fake_run(*a, **k):
        calls["n"] += 1
        return {"status": "completed"}

    monkeypatch.setattr(sched, "_config_ready", lambda: False)
    monkeypatch.setattr(sched, "run_sync", _fake_run)
    sched._auto_sync_job()
    assert calls["n"] == 0
    assert sched.get_state()["last_auto_sync"] is None


def test_trigger_sync_clears_running_if_enqueue_fails(monkeypatch):
    class _BadScheduler:
        def add_job(self, *a, **k):
            raise RuntimeError("enqueue boom")

    monkeypatch.setattr(sched, "_scheduler", _BadScheduler())
    res = sched.trigger_sync("folder")
    assert res["status"] == "error"
    assert sched.get_state()["running"] is False  # flag never left stuck


def test_auto_sync_runs_and_records_time(monkeypatch):
    monkeypatch.setattr(sched, "_config_ready", lambda: True)
    monkeypatch.setattr(
        sched, "run_sync", lambda *a, **k: {"status": "completed", "summary": {"added": 1}}
    )
    sched._auto_sync_job()
    state = sched.get_state()
    assert state["last_auto_sync"] is not None
    assert state["last_auto_summary"] == {"status": "completed", "summary": {"added": 1}}
    assert state["running"] is False


def test_auto_sync_skips_when_disabled(monkeypatch):
    """The user toggle gates the auto-sync job even when Drive is configured."""
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return {"status": "completed"}

    monkeypatch.setattr(sched, "_config_ready", lambda: True)
    monkeypatch.setattr(sched, "run_sync", _fake)
    sched._state["auto_sync_enabled"] = False
    sched._auto_sync_job()
    assert calls["n"] == 0
    assert sched.get_state()["auto_sync_enabled"] is False


def test_set_auto_sync_persists_to_config(temp_db):
    """Toggling persists to the config row and mirrors into the in-memory state."""
    sched.set_auto_sync(False)
    assert sched.get_state()["auto_sync_enabled"] is False
    s = dbs.SessionLocal()
    assert crud.get_or_create_config(s).auto_sync_enabled is False
    s.close()

    sched.set_auto_sync(True)
    assert sched.get_state()["auto_sync_enabled"] is True


def test_sync_resets_progress_to_none_when_done(monkeypatch):
    """When a run finishes, progress returns to None (idle contract) so the SSE
    stream / status don't leak the previous run's last per-file snapshot."""
    monkeypatch.setattr(sched, "_config_ready", lambda: True)
    monkeypatch.setattr(sched, "run_sync", lambda *a, **k: {"status": "completed", "summary": {}})
    sched._state["progress"] = {"done": 2, "total": 2, "file": "x.pdf", "phase": "embedding"}
    sched._auto_sync_job()
    assert sched.get_state()["progress"] is None
