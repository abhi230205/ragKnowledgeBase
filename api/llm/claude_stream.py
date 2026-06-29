"""Anthropic Messages API streaming + SSE re-emission.

`stream_answer` is an async generator of SSE event dicts consumed by
sse_starlette's EventSourceResponse. Normal event order: `token`* → `citations`
→ `done`. On a mid-stream failure it emits `citations` (if any tokens arrived) →
`error` → `done {interrupted: true}`. The no-context case streams the honest
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

# Shown when Claude returns a successful but content-free completion.
EMPTY_ANSWER_MESSAGE = "I wasn't able to generate an answer for that. Please try rephrasing."


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def error_stream(message: str):
    """Single-event SSE stream for failures that occur before/around streaming
    (e.g. retrieval errors in the route) — keeps everything on the SSE channel."""
    yield _sse("error", {"message": message})


async def _stream_claude_tokens(
    api_key: str, model: str, system: str, messages: list[dict], meta: dict | None = None
):
    """Async generator of text deltas from the Anthropic Messages API.

    Uses AsyncAnthropic (non-blocking); the SDK retries 429/transient errors with
    exponential backoff. If `meta` is given, the final message's stop_reason is
    recorded into it (so the caller can flag truncation).
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model, max_tokens=settings.max_output_tokens, system=system, messages=messages
    ) as stream:
        async for text in stream.text_stream:
            yield text
        if meta is not None:
            try:
                final = await stream.get_final_message()
                meta["stop_reason"] = getattr(final, "stop_reason", None)
            except Exception:  # pragma: no cover - usage/stop_reason is best-effort
                pass


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
    meta: dict = {}
    try:
        async for text in _stream_claude_tokens(api_key, model, system, messages, meta):
            parts.append(text)
            yield _sse("token", {"text": text})
    except Exception:  # rate limit after retries, timeout, mid-stream drop
        # Log full detail server-side; send the client a generic message (no
        # internal/provider detail leaks). Still emit citations for any partial
        # answer so its [n] markers map to sources, then a terminal `done`.
        logger.exception("Claude streaming failed")
        if parts:
            yield _sse("citations", citations)
        yield _sse("error", {"message": "The answer was interrupted. Please try again."})
        yield _sse("done", {"interrupted": True})
        if parts:
            await asyncio.to_thread(_persist, session_id, "".join(parts))
        return

    # Successful but empty completion: don't render a blank bubble with sources.
    if not parts:
        yield _sse("token", {"text": EMPTY_ANSWER_MESSAGE})
        yield _sse("citations", [])
        yield _sse("done", {"no_context": False, "empty": True})
        await asyncio.to_thread(_persist, session_id, EMPTY_ANSWER_MESSAGE)
        return

    truncated = meta.get("stop_reason") == "max_tokens"
    yield _sse("citations", citations)
    yield _sse("done", {"no_context": False, "truncated": truncated})
    await asyncio.to_thread(_persist, session_id, "".join(parts))
