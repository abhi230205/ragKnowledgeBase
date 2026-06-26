"""Search route (POST /search) — retrieval independent of chat, for testing.

TODO (Phase 3): embed the query with the same local model used for indexing,
run a top-k Chroma similarity search (optional file_id filter), normalise scores
(cosine: score = 1 - distance) and return source metadata.

Planned request:  {"query": "...", "top_k": 5, "file_id": null}
Planned response: {"query": "...", "results": [
    {"rank": 1, "score": 0.83, "file_name": "policy.pdf", "page": 4,
     "chunk_index": 11, "preview": "..."}]}
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["search"])


@router.post("/search")
def search():
    """Top-k similarity search. TODO: implement (Phase 3)."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "POST /search not implemented yet (Phase 3)"},
    )
