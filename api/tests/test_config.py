"""Tests for /config — secret masking, save round-trip, and validation.

Uses a throwaway temp DB (and nulls the env fallbacks) so these never read or
mutate the real config row / real secrets.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
from main import app
from routes import config as config_route

client = TestClient(app)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'cfg.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(config_route, "get_session", lambda: TestSession())
    # Null env fallbacks so masking reflects only the temp DB.
    monkeypatch.setattr(config_route.settings, "anthropic_api_key", "")
    monkeypatch.setattr(config_route.settings, "drive_folder_id", "")
    monkeypatch.setattr(
        config_route.settings, "google_service_account_path", str(tmp_path / "nope.json")
    )
    return TestSession


def test_mask_helper():
    assert config_route._mask(None) is None
    masked = config_route._mask("sk-ant-abcdefg12345")
    assert masked == "sk-ant…••••"
    assert "abcdefg" not in masked


def test_get_config_defaults(temp_db):
    data = client.get("/config").json()
    assert data["has_anthropic_key"] is False
    assert data["anthropic_key"] is None
    assert data["has_service_account"] is False
    assert data["top_k"] == 5


def test_post_saves_and_masks_secrets(temp_db):
    secret = "sk-ant-supersecret1234567890"
    r = client.post(
        "/config",
        json={"anthropic_api_key": secret, "top_k": 7, "drive_folder_id": "FOLDER123"},
    )
    assert r.status_code == 200
    body = r.json()
    # raw secret must never appear; key is masked; flags + values set
    assert secret not in json.dumps(body)
    assert body["anthropic_key"].endswith("••••")
    assert body["has_anthropic_key"] is True
    assert body["top_k"] == 7
    assert body["drive_folder_id"] == "FOLDER123"

    # persisted + still masked on GET
    g = client.get("/config").json()
    assert g["top_k"] == 7 and g["has_anthropic_key"] is True
    assert secret not in json.dumps(g)


def test_post_invalid_service_account_json_422(temp_db):
    assert client.post("/config", json={"service_account_json": "not json"}).status_code == 422


def test_post_service_account_missing_keys_422(temp_db):
    r = client.post("/config", json={"service_account_json": json.dumps({"foo": "bar"})})
    assert r.status_code == 422


def test_post_service_account_non_object_json_422(temp_db):
    # Valid JSON but not an object (array/number/null) must 422, not 500.
    for bad in ("[]", "123", "null", '"a string"'):
        assert client.post("/config", json={"service_account_json": bad}).status_code == 422


def test_post_top_k_must_be_positive_422(temp_db):
    assert client.post("/config", json={"top_k": 0}).status_code == 422


def test_blank_secret_leaves_existing_unchanged(temp_db):
    client.post("/config", json={"anthropic_api_key": "sk-ant-keep-me-123456"})
    # A later save with a blank key must not wipe the stored one.
    r = client.post("/config", json={"top_k": 3})
    assert r.json()["has_anthropic_key"] is True and r.json()["top_k"] == 3
