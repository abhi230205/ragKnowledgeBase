"""Embedded ChromaDB wrapper.

Uses chromadb.PersistentClient(path=settings.chroma_path) — a library call, not a
network service — with data on a mounted Docker volume so it survives restarts.
Collection uses cosine space (hnsw:space = "cosine"). Chunks are stored with
deterministic ids "{file_id}_{chunk_index}" so re-syncs are idempotent, plus
metadata: file_id, file_name, start_page, end_page, chunk_index, preview.

TODO (Phase 2/3):
- get_collection() -> persistent collection (created with cosine space).
- add(ids, embeddings, documents, metadatas)
- query(query_embedding, top_k, where=None) -> ids/documents/metadatas/distances
- delete(where={"file_id": ...})  (used on edit/delete to avoid orphan chunks)
- count() and reset() (the latter for an embedding-model change / full re-index).
"""

from __future__ import annotations


def get_collection():
    """Return the persistent Chroma collection. TODO: implement (Phase 2)."""
    raise NotImplementedError("chroma_store.get_collection — Phase 2")
