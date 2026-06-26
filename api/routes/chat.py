"""Chat route (POST /chat) — grounded, cited, token-streamed answers via SSE.

TODO (Phase 4):
- Retrieve top-k chunks for the (optionally history-rewritten) question.
- No-context guard: if best similarity < settings.relevance_threshold, stream the
  honest "not in the knowledge base" answer with an empty citations array.
- Build the grounding prompt: system (answer only from <context>, cite [n]) +
  trimmed history + a user turn carrying the <context> blocks with source/page.
- Stream Claude Messages API deltas and re-emit our own SSE events:
      event: token      data: {"text": "..."}
      event: citations  data: [{"id": 1, "file_name": "policy.pdf", "page": 4}]
      event: done       data: {"usage": {...}}
- Persist the turn to the session (SQLite) for multi-turn coherence.

Planned request: {"session_id": "s_42", "message": "...", "top_k": 5}
Response: text/event-stream (sse-starlette EventSourceResponse).
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat():
    """Stream a grounded, cited answer. TODO: implement (Phase 4)."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "POST /chat not implemented yet (Phase 4)"},
    )
