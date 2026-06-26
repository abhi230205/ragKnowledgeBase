"""Anthropic Messages API streaming + SSE re-emission.

`stream_answer` is an async generator of SSE event dicts consumed by
sse_starlette's EventSourceResponse. Event order: `token`* → `citations` → `done`
(or a single `error` event on failure). The no-context case streams the honest
fallback and makes NO Claude call. The Claude-token producer is isolated in
`_stream_claude_tokens` so tests can monkeypatch it.

The API key comes from the SQLite config (UI) or env settings — never hardcoded,
never logged.
"""

from __future__ import annotations

import asyncio
import json
import logging

from config import settings
from db import crud
from db.session import get_session
from llm import prompt_builder

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def _stream_claude_tokens(api_key: str, model: str, system: str, messages: list[dict]):
    """Async generator of text deltas from the Anthropic Messages API.

    Uses AsyncAnthropic (non-blocking). The SDK retries 429 / transient errors with
    exponential backoff automatically.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model, max_tokens=MAX_TOKENS, system=system, messages=messages
    ) as stream:
        async for text in stream.text_stream:
            yield text


def _persist(session_id: str, assistant_text: str) -> None:
    """Save the assistant turn so the session stays coherent. Best-effort."""
    if not session_id or not assistant_text:
        return
    session = get_session()
    try:
        crud.add_message(session, session_id, "assistant", assistant_text)
    except Exception:  # pragma: no cover - persistence is best-effort
        logger.exception("Failed to persist assistant message")
    finally:
        session.close()


async def stream_answer(
    session_id: str,
    question: str,
    chunks,
    history=None,
    api_key: str | None = None,
    model: str | None = None,
):
    """Yield SSE events for one chat turn. See module docstring for event order."""
    api_key = api_key or settings.anthropic_api_key
    model = model or settings.chat_model

    # No-context guard: honest fallback, empty citations, no Claude call.
    if not chunks:
        answer = prompt_builder.NO_CONTEXT_MESSAGE
        yield _sse("token", {"text": answer})
        yield _sse("citations", [])
        yield _sse("done", {"no_context": True})
        await asyncio.to_thread(_persist, session_id, answer)
        return

    system, messages, citations = prompt_builder.build_messages(question, chunks, history)

    if not api_key:
        yield _sse("error", {"message": "Anthropic API key is not configured."})
        return

    parts: list[str] = []
    try:
        async for text in _stream_claude_tokens(api_key, model, system, messages):
            parts.append(text)
            yield _sse("token", {"text": text})
    except Exception:  # rate limit after retries, timeout, mid-stream drop
        # Log the full error server-side; send the client a generic message (no
        # internal/provider detail leaks to the browser).
        logger.exception("Claude streaming failed")
        yield _sse("error", {"message": "The answer was interrupted. Please try again."})
        if parts:
            await asyncio.to_thread(_persist, session_id, "".join(parts))
        return

    yield _sse("citations", citations)
    yield _sse("done", {"no_context": False})
    await asyncio.to_thread(_persist, session_id, "".join(parts))
