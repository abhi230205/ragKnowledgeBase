"""Chat route (POST /chat) — grounded, cited, token-streamed answers via SSE.

Flow: validate → load windowed history → record the user turn → retrieve top-k
(off the event loop) → apply the no-context threshold guard → stream the answer
as SSE (`token`* → `citations` → `done`, or `error`). Credentials/model come from
the SQLite config (UI) or env settings — never hardcoded.

Request: {"session_id": "s_42", "message": "...", "top_k": 5}
Response: text/event-stream
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from db import crud
from db.session import get_session
from embeddings import embedder
from llm import claude_stream
from vectorstore import chroma_store

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = None


def _retrieve(query: str, top_k: int) -> list[dict]:
    """Embed + top-k search + no-context threshold guard. Runs in a threadpool.

    Returns [] (→ no-context path) when the KB is empty or the best hit is below
    the relevance floor, so junk context is never handed to Claude.
    """
    collection = chroma_store.get_collection()
    if chroma_store.count(collection) == 0:
        return []
    hits = chroma_store.query(collection, embedder.embed_query(query), top_k=top_k)
    if not hits or hits[0]["score"] < settings.relevance_threshold:
        return []
    return hits


@router.post("/chat")
async def chat(body: ChatRequest):
    question = (body.message or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="message must not be empty")
    if not (body.session_id or "").strip():
        raise HTTPException(status_code=422, detail="session_id is required")

    top_k = body.top_k if body.top_k and body.top_k > 0 else settings.top_k

    # Resolve creds/model + history; record the user turn (before streaming).
    session = get_session()
    try:
        cfg = crud.get_or_create_config(session)
        api_key = cfg.anthropic_api_key or settings.anthropic_api_key
        model = cfg.chat_model or settings.chat_model
        history = crud.get_history(session, body.session_id, settings.max_history_turns * 2)
        crud.add_message(session, body.session_id, "user", question)
    finally:
        session.close()

    chunks = await run_in_threadpool(_retrieve, question, top_k)

    generator = claude_stream.stream_answer(
        body.session_id, question, chunks, history, api_key=api_key, model=model
    )
    return EventSourceResponse(generator)
