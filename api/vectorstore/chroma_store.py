"""Embedded ChromaDB wrapper.

chromadb.PersistentClient(path=settings.chroma_path) — a library call, not a
network service — with data on a mounted volume so it survives restarts. The
collection uses cosine space; we always pass our own embeddings (embedding_function
is None) so Chroma never downloads its default ONNX model.

Chunk ids are deterministic ("{file_id}_{chunk_index}") so re-syncs are
idempotent (upsert), and per-chunk metadata carries file_id, file_name,
start_page, end_page, chunk_index, preview.
"""

from __future__ import annotations

import logging
import threading

from config import settings

logger = logging.getLogger(__name__)

_clients: dict[str, object] = {}
_lock = threading.Lock()

_PREVIEW_CHARS = 160


def get_client(path: str | None = None):
    """Return a cached PersistentClient for the given path (default: settings)."""
    import chromadb

    key = path or settings.chroma_path
    client = _clients.get(key)
    if client is None:
        with _lock:
            client = _clients.get(key)
            if client is None:
                client = chromadb.PersistentClient(path=key)
                _clients[key] = client
    return client


def get_collection(client=None, name: str | None = None):
    """Get-or-create the knowledge-base collection (cosine, BYO embeddings).

    `name` overrides the default collection name (used by tests for isolation).
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=name or settings.collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def add_chunks(collection, file_id: str, file_name: str, chunks, embeddings) -> int:
    """Upsert a file's chunks + embeddings. Returns the number added."""
    if not chunks:
        return 0
    ids = [f"{file_id}_{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "file_id": file_id,
            "file_name": file_name,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "chunk_index": c.chunk_index,
            "preview": c.text[:_PREVIEW_CHARS],
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def query(collection, query_embedding: list[float], top_k: int = 5, where: dict | None = None):
    """Top-k similarity search. Returns ranked dicts with score = 1 - distance."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        meta = meta or {}
        out.append(
            {
                "rank": rank,
                "score": round(1.0 - float(dist), 4),
                "distance": float(dist),
                "chunk_id": cid,
                "file_id": meta.get("file_id"),
                "file_name": meta.get("file_name"),
                "page": meta.get("start_page"),
                "start_page": meta.get("start_page"),
                "end_page": meta.get("end_page"),
                "chunk_index": meta.get("chunk_index"),
                "preview": meta.get("preview"),
                "document": doc,
            }
        )
    return out


def delete_file(collection, file_id: str) -> None:
    """Delete all chunks for a file (used on edit/delete to avoid orphans)."""
    collection.delete(where={"file_id": file_id})


def update_file_name(collection, file_id: str, new_name: str) -> int:
    """Update file_name metadata for a renamed file (no re-embedding). Returns count."""
    existing = collection.get(where={"file_id": file_id}, include=["metadatas"])
    ids = existing.get("ids") or []
    if not ids:
        return 0
    metas = existing.get("metadatas") or []
    for m in metas:
        m["file_name"] = new_name
    collection.update(ids=ids, metadatas=metas)
    return len(ids)


def count(collection) -> int:
    """Total number of chunks in the collection."""
    return collection.count()


def reset_collection(client=None) -> None:
    """Drop and recreate the collection (e.g. after an embedding-model change)."""
    client = client or get_client()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:  # pragma: no cover - collection may not exist yet
        pass
    get_collection(client)


def _stamp(collection, model: str, dim: int) -> None:
    # NOTE: do NOT include "hnsw:space" here — Chroma rejects any modify() carrying
    # the distance function (it's fixed at creation), which would make the whole
    # stamp fail. The space stays cosine from get_collection()'s create call.
    try:
        collection.modify(metadata={"embedding_model": model, "embedding_dim": dim})
    except Exception:  # pragma: no cover - metadata stamp is best-effort
        logger.warning("Could not stamp collection embedding metadata")


def ensure_model(client, model: str, dim: int) -> bool:
    """Ensure the collection matches the active embedding model/dimension.

    Stamps the model + dim into the collection metadata. If a *different* model/dim
    was previously recorded, the collection is reset (dropped + recreated) so the
    corpus is re-embedded with the new model — instead of a silent per-file
    dimension-mismatch error storm. Returns True iff a reset happened.
    Call this at sync start (it does not load the model itself; the caller passes dim).
    """
    coll = get_collection(client)
    meta = coll.metadata or {}
    cur_model, cur_dim = meta.get("embedding_model"), meta.get("embedding_dim")

    if cur_model == model and cur_dim == dim:
        return False
    if cur_model is None:  # not stamped yet (fresh / legacy) — stamp, no reset
        _stamp(coll, model, dim)
        return False
    # A different model/dim was recorded -> reset and re-stamp.
    reset_collection(client)
    _stamp(get_collection(client), model, dim)
    return True
