"""Smoke test: /health returns 200. Wires pytest from Day 1.

The full ≥10-test suite (chunking, embedding, retrieval, sync diff, chat shape)
lands across Phases 2–6 per the test plan.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
