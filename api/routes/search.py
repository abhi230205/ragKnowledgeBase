"""Search route (POST /search) — retrieval independent of chat, for testing.

Embeds the query with the same local model used for indexing, runs a top-k cosine
search in Chroma (optional file_id filter), and returns scored results with source
metadata. Defined as a sync route so FastAPI runs the CPU-bound embed/query call in
its threadpool (off the event loop).

Request:  {"query": "...", "top_k": 5, "file_id": null}
Response: {"query": "...", "count": n, "results": [
    {"rank": 1, "score": 0.83, "file_name": "policy.pdf", "page": 4,
     "chunk_index": 11, "preview": "..."}], "relevance_threshold": 0.25}

`relevance_threshold` echoes the configured cosine floor (informational only — this
endpoint returns all top-k hits so /rag-eval can see below-floor scores; the
no-context guard that actually applies the floor lives in /chat).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from embeddings import embedder
from vectorstore import chroma_store

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    file_id: Optional[str] = None


@router.post("/search")
def search(body: SearchRequest) -> dict:
    """Top-k similarity search over the knowledge base."""
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    top_k = body.top_k if body.top_k and body.top_k > 0 else settings.top_k

    collection = chroma_store.get_collection()
    if chroma_store.count(collection) == 0:
        return {
            "query": query,
            "count": 0,
            "results": [],
            "relevance_threshold": settings.relevance_threshold,
        }

    where = {"file_id": body.file_id} if body.file_id else None
    query_vec = embedder.embed_query(query)
    hits = chroma_store.query(collection, query_vec, top_k=top_k, where=where)

    results = [
        {
            "rank": h["rank"],
            "score": h["score"],
            "file_name": h["file_name"],
            "page": h["page"],
            "start_page": h["start_page"],
            "end_page": h["end_page"],
            "chunk_index": h["chunk_index"],
            "preview": h["preview"],
        }
        for h in hits
    ]
    return {
        "query": query,
        "count": len(results),
        "results": results,
        "threshold": settings.relevance_threshold,
    }
