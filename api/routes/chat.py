"""Chat route (POST /chat) — grounded, cited, token-streamed answers via SSE.

Flow: validate → load windowed history → record the user turn → retrieve top-k
(off the event loop) → apply the no-context threshold guard → stream the answer
as SSE (`token`* → `citations` → `done`, or `error`). Credentials/model come from
the SQLite config (UI) or env settings — never hardcoded.

Request: {"session_id": "s_42", "message": "...", "top_k": 5}
Response: text/event-stream
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from db import crud
from db.session import get_session
from embeddings import embedder, reranker
from llm import claude_stream
from vectorstore import chroma_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = None


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Embed + top-N search + no-context threshold guard + cross-encoder re-rank.
    Runs in a threadpool.

    Returns [] (→ no-context path) when the KB is empty or the best cosine hit is
    below the relevance floor, so junk context is never handed to Claude. The
    threshold guard stays on the cosine score (the "is anything relevant" decision);
    re-ranking then reorders the surviving candidates and keeps the best top_k.
    """
    collection = chroma_store.get_collection()
    if chroma_store.count(collection) == 0:
        return []
    # Pull a wider candidate set when re-ranking is on, then let the cross-encoder pick.
    candidates = chroma_store.query(
        collection, embedder.embed_query(query), top_k=reranker.candidate_count(top_k)
    )
    if not candidates or candidates[0]["score"] < settings.relevance_threshold:
        return []
    return reranker.rerank(query, candidates, top_k)


@router.post("/chat")
async def chat(body: ChatRequest):
    question = (body.message or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="message must not be empty")
    if not (body.session_id or "").strip():
        raise HTTPException(status_code=422, detail="session_id is required")

    # Resolve creds/model/top_k + load windowed history (BEFORE recording this turn).
    session = get_session()
    try:
        cfg = crud.get_or_create_config(session)
        api_key = cfg.anthropic_api_key or settings.anthropic_api_key
        model = cfg.chat_model or settings.chat_model
        # Honor the request, then the UI-configured top_k, then the env default.
        top_k = body.top_k if (body.top_k and body.top_k > 0) else (cfg.top_k or settings.top_k)
        history = crud.get_history(session, body.session_id, settings.max_history_turns * 2)
    finally:
        session.close()

    # Retrieve off the event loop. A failure here streams a graceful SSE error
    # (never a 500) and records no user turn, so history stays consistent.
    try:
        chunks = await run_in_threadpool(_retrieve, question, top_k)
    except Exception:
        logger.exception("Retrieval failed for /chat")
        return EventSourceResponse(
            claude_stream.error_stream("Could not search the knowledge base. Please try again.")
        )

    # Record the user turn only after retrieval succeeds.
    session = get_session()
    try:
        crud.add_message(session, body.session_id, "user", question)
    finally:
        session.close()

    generator = claude_stream.stream_answer(
        body.session_id, question, chunks, history, api_key=api_key, model=model
    )
    return EventSourceResponse(generator)
