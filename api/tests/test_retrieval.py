"""Retrieval tests — relevance, top_k, and file_id filtering.

Uses the real embedder + an in-memory (ephemeral) Chroma collection, so it
exercises the actual add/query path without touching the persistent volume. Run
in the container (loads the model): `docker compose exec api pytest`.
"""

from __future__ import annotations

import uuid

import chromadb
from embeddings import embedder
from ingestion.chunker import Chunk
from vectorstore import chroma_store


def _chunk(text: str, idx: int, page: int = 1) -> Chunk:
    return Chunk(text=text, chunk_index=idx, start_page=page, end_page=page)


def _fresh_collection():
    # EphemeralClient is a process-singleton, so use a unique collection name per
    # test for isolation (absolute counts would otherwise accumulate across tests).
    return chroma_store.get_collection(
        client=chromadb.EphemeralClient(), name=f"test_{uuid.uuid4().hex}"
    )


def test_retrieval_returns_relevant_chunk():
    coll = _fresh_collection()
    chunks = [
        _chunk("Refunds are issued within 14 days of delivery.", 0),
        _chunk("Our headquarters is located in Bangalore, India.", 1),
        _chunk("Photosynthesis converts sunlight into chemical energy in plants.", 2),
    ]
    embs = embedder.embed_texts([c.text for c in chunks])
    added = chroma_store.add_chunks(coll, "fileA", "policy.pdf", chunks, embs)
    assert added == 3

    q = embedder.embed_query("What is the refund window for returns?")
    results = chroma_store.query(coll, q, top_k=3)

    assert results[0]["chunk_index"] == 0  # refund chunk ranks first
    assert results[0]["file_name"] == "policy.pdf"
    assert 0.0 <= results[0]["score"] <= 1.0001  # cosine score = 1 - distance


def test_topk_and_file_filter():
    coll = _fresh_collection()
    a = [_chunk("Alpha document discussing feline behaviour and cats.", 0)]
    b = [_chunk("Beta document discussing canine behaviour and dogs.", 0)]
    chroma_store.add_chunks(coll, "A", "a.pdf", a, embedder.embed_texts([a[0].text]))
    chroma_store.add_chunks(coll, "B", "b.pdf", b, embedder.embed_texts([b[0].text]))

    q = embedder.embed_query("animals")
    assert len(chroma_store.query(coll, q, top_k=1)) == 1  # respects top_k

    scoped = chroma_store.query(coll, q, top_k=5, where={"file_id": "B"})
    assert scoped and all(r["file_id"] == "B" for r in scoped)  # filter works


def test_delete_file_removes_chunks():
    coll = _fresh_collection()
    chunks = [_chunk("Some content to be deleted.", 0)]
    chroma_store.add_chunks(coll, "DEL", "del.pdf", chunks, embedder.embed_texts([chunks[0].text]))
    assert chroma_store.count(coll) == 1
    chroma_store.delete_file(coll, "DEL")
    assert chroma_store.count(coll) == 0
