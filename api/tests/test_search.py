"""Integration tests for POST /search — route + retrieval over a seeded collection.

Patches chroma_store.get_collection to an in-memory ephemeral collection so the
route's real embed → query → response path is exercised without a sync.
"""

from __future__ import annotations

import uuid

import chromadb
from fastapi.testclient import TestClient

from embeddings import embedder
from ingestion.chunker import Chunk
from main import app
from vectorstore import chroma_store

client = TestClient(app)


def _seed(*docs):
    """docs: (file_id, file_name, [texts]). Returns a seeded ephemeral collection."""
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


def test_search_returns_relevant(monkeypatch):
    coll = _seed(
        (
            "A",
            "policy.pdf",
            [
                "Refunds are issued within 14 days of delivery.",
                "Shipping is free over $50.",
            ],
        ),
        ("B", "about.pdf", ["The company is headquartered in Bangalore, India."]),
    )
    monkeypatch.setattr(chroma_store, "get_collection", lambda *a, **k: coll)

    r = client.post("/search", json={"query": "what is the refund window?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    assert data["results"][0]["file_name"] == "policy.pdf"
    assert "preview" in data["results"][0] and "page" in data["results"][0]


def test_search_respects_top_k_and_filter(monkeypatch):
    coll = _seed(
        ("A", "a.pdf", ["Alpha content about cats and feline behaviour."]),
        ("B", "b.pdf", ["Beta content about dogs and canine behaviour."]),
    )
    monkeypatch.setattr(chroma_store, "get_collection", lambda *a, **k: coll)

    assert client.post("/search", json={"query": "animals", "top_k": 1}).json()["count"] == 1

    res = client.post("/search", json={"query": "animals", "top_k": 5, "file_id": "B"}).json()[
        "results"
    ]
    assert res and all(x["file_name"] == "b.pdf" for x in res)


def test_search_empty_query_422():
    assert client.post("/search", json={"query": "   "}).status_code == 422


def test_search_empty_collection(monkeypatch):
    empty = chroma_store.get_collection(
        client=chromadb.EphemeralClient(), name=f"t_{uuid.uuid4().hex}"
    )
    monkeypatch.setattr(chroma_store, "get_collection", lambda *a, **k: empty)
    data = client.post("/search", json={"query": "anything"}).json()
    assert data["count"] == 0 and data["results"] == []
