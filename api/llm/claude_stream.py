"""Anthropic Messages API streaming + SSE re-emission.

TODO (Phase 4):
- Use the Anthropic SDK: `with client.messages.stream(model=settings.chat_model,
  max_tokens=..., system=..., messages=...) as stream:` and iterate
  `stream.text_stream` for text deltas.
- Re-emit our own SSE events to the browser: token -> citations -> done.
- Handle 429 (rate limit) with exponential backoff; on a mid-stream timeout/error
  emit an `error` event so the UI can show "response interrupted" gracefully.

Note: the API key comes from the SQLite config (UI-entered) or settings, never
hardcoded.
"""

from __future__ import annotations


async def stream_answer(messages, system: str):
    """Async generator yielding SSE event payloads. TODO: implement (Phase 4)."""
    raise NotImplementedError("claude_stream.stream_answer — Phase 4")
