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
