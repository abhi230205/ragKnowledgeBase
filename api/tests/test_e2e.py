"""End-to-end tests: the full lifecycle wired together, faking only the external
boundaries (Google Drive + the Anthropic stream).

REAL components exercised: pdf_parser (real PDFs via PyMuPDF), chunker, the
sentence-transformers embedder, embedded Chroma, SQLite, the ingestion pipeline,
and the /search, /chat, /status routes. FAKED: Google Drive (in-memory PDFs) and
the Anthropic token stream (canned deltas).

Each test runs in an isolated temp SQLite + temp Chroma, so it never touches the
live knowledge base. `get_session` resolves `db.session.SessionLocal` at call time,
so patching that module global redirects every component (pipeline, routes, crud)
to the temp DB at once.
"""

from __future__ import annotations

import json

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import db.session as dbs
import vectorstore.chroma_store as cs
from config import settings
from db.models import Base
from ingestion import pipeline
from ingestion.drive_client import PDF_MIME, DriveFile
from llm import claude_stream, prompt_builder
from main import app

client = TestClient(app)


# ---------------------------------------------------------------- helpers


def _pdf(pages: list[tuple[str, str]]) -> bytes:
    doc = fitz.open()
    for title, body in pages:
        p = doc.new_page()
        p.insert_textbox(fitz.Rect(72, 72, 523, 120), title, fontsize=14, fontname="helv")
        p.insert_textbox(fitz.Rect(72, 135, 523, 760), body, fontsize=11, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf() -> bytes:
    """Image-only PDF (no text layer) -> should be flagged no_extractable_text."""
    src = fitz.open()
    sp = src.new_page()
    sp.insert_textbox(fitz.Rect(60, 80, 535, 400), "Scan only - no text layer", fontsize=14)
    pix = sp.get_pixmap(dpi=100)
    img = fitz.open()
    ip = img.new_page(width=sp.rect.width, height=sp.rect.height)
    ip.insert_image(ip.rect, pixmap=pix)
    data = img.tobytes()
    img.close()
    src.close()
    return data


def _dfile(fid: str, name: str, md5: str) -> DriveFile:
    return DriveFile(
        id=fid,
        name=name,
        mime_type=PDF_MIME,
        md5_checksum=md5,
        modified_time="2026-01-01T00:00:00Z",
        size=1234,
    )


class _FakeDrive:
    """Stand-in for ingestion.drive_client.DriveClient over an in-memory corpus."""

    def __init__(self, corpus):  # corpus: list[(DriveFile, bytes)]
        self._corpus = corpus

    def list_pdfs(self, folder_id, recursive=True):
        return [f for f, _ in self._corpus]

    def download_bytes(self, file_id):
        return next(b for f, b in self._corpus if f.id == file_id)


def _use_corpus(monkeypatch, corpus):
    monkeypatch.setattr(pipeline, "DriveClient", lambda **kw: _FakeDrive(corpus))


def _default_corpus():
    return [
        (
            _dfile("f_refund", "acme_refund_policy.pdf", "md5-refund-v1"),
            _pdf(
                [
                    (
                        "Refund Policy",
                        "Refunds are issued within 14 days of delivery, no questions asked. "
                        "Standard shipping is free on all orders over $50.",
                    )
                ]
            ),
        ),
        (
            _dfile("f_handbook", "acme_handbook.pdf", "md5-handbook-v1"),
            _pdf(
                [
                    ("Handbook - Leave", "Employees accrue 20 days of paid annual leave per year."),
                    ("Handbook - Remote", "Employees may work remotely up to three days per week."),
                ]
            ),
        ),
        (_dfile("f_scanned", "acme_scanned.pdf", "md5-scanned-v1"), _scanned_pdf()),
    ]


@pytest.fixture
def e2e_env(tmp_path, monkeypatch):
    # Isolated temp SQLite — redirect get_session everywhere via the module global.
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'e2e.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(dbs, "engine", engine)
    monkeypatch.setattr(dbs, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    # Isolated temp Chroma — fresh client cache at a temp path.
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(cs, "_clients", {})
    yield
    cs._clients.clear()


def _sse_events(text: str):
    """Parse an SSE response body into (event, data) pairs."""
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


# ---------------------------------------------------------------- tests


def test_e2e_sync_ingests_embeds_and_flags_scanned(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    result = pipeline.run_sync("folder-id")

    assert result["status"] == "completed"
    s = result["summary"]
    assert s["added"] == 2 and s["unchanged"] == 0  # refund + handbook embedded
    assert s["chunks_created"] >= 2
    assert any(e["reason"] == "no_extractable_text" for e in s["errors"])  # scanned flagged

    status = client.get("/status").json()
    assert status["documents"] == 2
    assert status["chunks"] >= 2
    assert any(e["file"] == "acme_scanned.pdf" for e in status["errors"])


def test_e2e_search_returns_relevant(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    res = client.post("/search", json={"query": "what is the refund window?", "top_k": 3}).json()
    assert res["count"] >= 1
    assert res["results"][0]["file_name"] == "acme_refund_policy.pdf"


def test_e2e_chat_grounded_with_citations(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")

    # Precondition: retrieval must clear the relevance floor; otherwise /chat would
    # silently take the no-context path and the citation assertion would mislead.
    pre = client.post("/search", json={"query": "What is the refund window?", "top_k": 3}).json()
    assert pre["count"] >= 1
    assert pre["results"][0]["score"] >= settings.relevance_threshold

    async def fake(api_key, model, system, messages, meta=None):
        for t in ["Refunds are issued within 14 days. ", "[1]"]:
            yield t

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", fake)
    monkeypatch.setattr(claude_stream.settings, "anthropic_api_key", "test-key")

    r = client.post("/chat", json={"session_id": "e2e", "message": "What is the refund window?"})
    assert r.status_code == 200
    events = _sse_events(r.text)
    kinds = [e for e, _ in events]
    assert "token" in kinds and kinds[-1] == "done"
    cites = next(d for e, d in events if e == "citations")
    assert any(c["file_name"] == "acme_refund_policy.pdf" for c in cites)


def test_e2e_chat_no_context_refuses(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    # Force the no-context path deterministically (no Claude call is made).
    monkeypatch.setattr(settings, "relevance_threshold", 0.99)
    r = client.post("/chat", json={"session_id": "e2e-nc", "message": "Who is the CEO?"})
    assert r.status_code == 200
    events = _sse_events(r.text)
    assert events[0][1]["text"] == prompt_builder.NO_CONTEXT_MESSAGE
    assert any(e == "citations" and d == [] for e, d in events)


def test_e2e_resync_skips_unchanged(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    s2 = pipeline.run_sync("folder-id")["summary"]
    # All three tracked files (incl. the flagged scan) are unchanged on re-sync.
    assert s2["added"] == 0
    assert s2["unchanged"] == 3
    assert s2["chunks_created"] == 0


def test_e2e_modified_file_reembeds(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")

    corpus = _default_corpus()
    corpus[0] = (
        _dfile("f_refund", "acme_refund_policy.pdf", "md5-refund-v2"),  # new checksum
        _pdf([("Refund Policy", "Refunds now take 30 business days to process.")]),
    )
    _use_corpus(monkeypatch, corpus)
    s = pipeline.run_sync("folder-id")["summary"]
    assert s["modified"] == 1

    res = client.post("/search", json={"query": "how long do refunds take now?"}).json()
    assert res["results"][0]["file_name"] == "acme_refund_policy.pdf"


def test_e2e_deleted_file_removes_chunks(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    before = client.get("/status").json()["documents"]

    corpus = [c for c in _default_corpus() if c[0].id != "f_handbook"]  # drop handbook
    _use_corpus(monkeypatch, corpus)
    s = pipeline.run_sync("folder-id")["summary"]
    assert s["deleted"] == 1

    after = client.get("/status").json()
    assert after["documents"] == before - 1


def test_e2e_renamed_file_updates_metadata_no_reembed(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    before = client.get("/status").json()

    corpus = _default_corpus()
    # Same id + same md5, only the name changes -> renamed (metadata-only).
    corpus[0] = (_dfile("f_refund", "acme_refund_policy_v2.pdf", "md5-refund-v1"), corpus[0][1])
    _use_corpus(monkeypatch, corpus)
    s = pipeline.run_sync("folder-id")["summary"]

    assert s["renamed"] == 1 and s["modified"] == 0 and s["added"] == 0
    assert s["chunks_created"] == 0  # no re-embedding on a pure rename
    res = client.post("/search", json={"query": "refund window"}).json()
    assert res["results"][0]["file_name"] == "acme_refund_policy_v2.pdf"  # new name in Chroma
    after = client.get("/status").json()
    assert after["documents"] == before["documents"] and after["chunks"] == before["chunks"]


def test_e2e_sync_preflight_422_when_unconfigured(e2e_env, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "drive_folder_id", None)
    monkeypatch.setattr(settings, "google_service_account_path", str(tmp_path / "nope.json"))

    # No folder configured -> 422 (folder check fires first).
    r = client.post("/sync", json={})
    assert r.status_code == 422 and "folder" in r.json()["error"].lower()

    # Folder set but still no credentials -> 422.
    client.post("/config", json={"drive_folder_id": "FOLDER123"})
    r = client.post("/sync", json={})
    assert r.status_code == 422 and "service-account" in r.json()["error"].lower()


def test_e2e_sync_auth_failure_returns_error_dict(e2e_env, monkeypatch, tmp_path):
    # Real DriveClient, no creds configured -> graceful error dict, never a raise/500.
    monkeypatch.setattr(settings, "google_service_account_path", str(tmp_path / "nope.json"))
    result = pipeline.run_sync("folder-id")
    assert result["status"] == "error"
    assert "credential" in result["error"].lower()
    assert client.get("/status").json()["documents"] == 0


def test_e2e_chat_midstream_error_over_http(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")

    async def boom(api_key, model, system, messages, meta=None):
        yield "Refunds are issued [1]"
        raise RuntimeError("connection dropped")

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", boom)
    monkeypatch.setattr(claude_stream.settings, "anthropic_api_key", "test-key")

    r = client.post(
        "/chat", json={"session_id": "e2e-mid", "message": "What is the refund window?"}
    )
    assert r.status_code == 200  # SSE stays 200 — a mid-stream raise never becomes a 500
    events = _sse_events(r.text)
    kinds = [e for e, _ in events]
    assert "token" in kinds and "citations" in kinds and "error" in kinds
    assert kinds[-1] == "done" and events[-1][1].get("interrupted") is True


def test_e2e_chat_multiturn_replays_history(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")

    captured: list = []

    async def fake(api_key, model, system, messages, meta=None):
        captured.append(list(messages))
        for t in ["Refunds are issued within 14 days. ", "[1]"]:
            yield t

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", fake)
    monkeypatch.setattr(claude_stream.settings, "anthropic_api_key", "test-key")

    client.post("/chat", json={"session_id": "s_mt", "message": "What is the refund window?"})
    client.post("/chat", json={"session_id": "s_mt", "message": "Is shipping free?"})

    assert len(captured) == 2
    assert len(captured[1]) > len(captured[0])  # turn 2 replays prior turns
    flat = " ".join(m["content"] for m in captured[1])
    assert "What is the refund window?" in flat  # turn-1 user question
    assert "Refunds are issued within 14 days." in flat  # turn-1 assistant answer
    assert captured[1][-1]["role"] == "user" and "shipping" in captured[1][-1]["content"].lower()


def test_e2e_data_persists_across_client_reopen(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")
    before = client.get("/status").json()

    # Simulate a restart: drop the cached Chroma client; the next call re-opens the
    # same on-disk path. (SQLite is already file-backed in the temp env.)
    cs._clients.clear()

    coll = cs.get_collection()
    assert cs.count(coll) == before["chunks"]  # vectors read back from disk
    res = client.post("/search", json={"query": "what is the refund window?", "top_k": 3}).json()
    assert res["results"][0]["file_name"] == "acme_refund_policy.pdf"
    after = client.get("/status").json()
    assert after["documents"] == before["documents"] and after["chunks"] == before["chunks"]


def test_e2e_embedding_model_change_triggers_reindex(e2e_env, monkeypatch):
    _use_corpus(monkeypatch, _default_corpus())
    pipeline.run_sync("folder-id")  # stamps the collection with the real model/dim

    # Fake a collection previously indexed with a different model, then re-sync.
    cs.get_collection().modify(metadata={"embedding_model": "old-model", "embedding_dim": 384})
    s = pipeline.run_sync("folder-id")["summary"]

    assert s["reindexed"] is True
    assert s["added"] == 2 and s["chunks_created"] >= 2  # whole corpus re-embedded
    assert client.get("/status").json()["documents"] == 2


def _table_pdf() -> bytes:
    """A bordered table PDF (ruled grid) so PyMuPDF find_tables() + pdfplumber fire."""
    doc = fitz.open()
    p = doc.new_page()
    cols_x = [72, 250, 420]
    rows_y = [120 + 30 * i for i in range(4)]
    sh = p.new_shape()
    for x in cols_x:
        sh.draw_line((x, rows_y[0]), (x, rows_y[-1]))
    for y in rows_y:
        sh.draw_line((cols_x[0], y), (cols_x[-1], y))
    sh.finish(width=0.7)
    sh.commit()
    rows = [["Plan", "Price"], ["Pro", "$99 per month"], ["Basic", "$9 per month"]]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            p.insert_text((cols_x[j] + 6, rows_y[i] + 20), cell, fontsize=11, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


def test_e2e_table_pdf_extracted_and_searchable(e2e_env, monkeypatch):
    corpus = [(_dfile("f_pricing", "acme_pricing.pdf", "md5-pricing-v1"), _table_pdf())]
    _use_corpus(monkeypatch, corpus)
    s = pipeline.run_sync("folder-id")["summary"]
    assert s["added"] == 1

    res = client.post("/search", json={"query": "Pro plan price per month", "top_k": 3}).json()
    assert res["count"] >= 1
    assert res["results"][0]["file_name"] == "acme_pricing.pdf"
    assert "99" in (res["results"][0]["preview"] or "")  # table cell text was extracted
