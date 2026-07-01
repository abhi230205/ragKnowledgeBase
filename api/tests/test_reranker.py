"""Tests for the cross-encoder re-ranking bonus.

These flip settings.rerank_enabled ON (conftest gates it off for the rest of the
suite), so the real CrossEncoder loads here. The key behaviour: given candidates
in a deliberately wrong (cosine) order, the cross-encoder pulls the genuinely
relevant chunk to the top — that's the retrieval-quality win the bonus buys.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient

from config import settings
from embeddings import embedder, reranker
from ingestion.chunker import Chunk
from main import app
from vectorstore import chroma_store

client = TestClient(app)


def test_candidate_count_widens_only_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "rerank_candidates", 20)
    monkeypatch.setattr(settings, "rerank_enabled", True)
    assert reranker.candidate_count(5) == 20
    assert reranker.candidate_count(30) == 30  # never shrinks below the requested top_k
    monkeypatch.setattr(settings, "rerank_enabled", False)
    assert reranker.candidate_count(5) == 5  # off => plain top_k


def test_rerank_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    cands = [
        {"document": "x", "score": 0.9, "rank": 1},
        {"document": "y", "score": 0.8, "rank": 2},
        {"document": "z", "score": 0.7, "rank": 3},
    ]
    out = reranker.rerank("q", [dict(c) for c in cands], top_k=2)
    assert [c["document"] for c in out] == ["x", "y"]  # original order, truncated
    assert "rerank_score" not in out[0]
    assert out[0]["rank"] == 1 and out[1]["rank"] == 2


def test_rerank_reorders_by_relevance(monkeypatch):
    """Cross-encoder promotes the relevant chunk even when it was ranked last."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    query = "How many days do I have to return a product for a refund?"
    candidates = [
        {
            "document": "The office cafeteria serves lunch from noon to 2pm on weekdays.",
            "score": 0.55,
            "rank": 1,
            "file_name": "misc.pdf",
            "chunk_index": 0,
        },
        {
            "document": "Refunds are issued within 14 days of delivery, no questions asked.",
            "score": 0.42,
            "rank": 2,
            "file_name": "policy.pdf",
            "chunk_index": 1,
        },
    ]
    out = reranker.rerank(query, [dict(c) for c in candidates], top_k=2)
    assert out[0]["document"].startswith("Refunds")  # relevant chunk promoted to #1
    assert out[0]["rank"] == 1
    assert "rerank_score" in out[0] and out[0]["rerank_score"] > out[1]["rerank_score"]


def _seed(*docs):
    coll = chroma_store.get_collection(
        client=chromadb.EphemeralClient(), name=f"t_{uuid.uuid4().hex}"
    )
    for file_id, file_name, texts in docs:
        chunks = [
            Chunk(text=t, chunk_index=i, start_page=i + 1, end_page=i + 1)
            for i, t in enumerate(texts)
        ]
        embs = embedder.embed_texts([c.text for c in chunks])
        chroma_store.add_chunks(coll, file_id, file_name, chunks, embs)
    return coll


def test_search_endpoint_applies_reranking(monkeypatch):
    """End-to-end: /search returns the relevant doc top with a rerank_score attached."""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    coll = _seed(
        ("A", "policy.pdf", ["Refunds are issued within 14 days of delivery."]),
        ("B", "menu.pdf", ["The cafeteria serves lunch from noon to 2pm."]),
    )
    monkeypatch.setattr(chroma_store, "get_collection", lambda *a, **k: coll)

    data = client.post("/search", json={"query": "refund window in days", "top_k": 2}).json()
    assert data["count"] >= 1
    assert data["results"][0]["file_name"] == "policy.pdf"
    assert data["results"][0]["rerank_score"] is not None  # re-ranking actually ran
