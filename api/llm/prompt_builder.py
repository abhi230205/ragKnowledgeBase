"""Grounding prompt construction for the RAG chat (the 20-point criterion).

TODO (Phase 4):
- System prompt: answer ONLY from <context>; if the answer isn't there, say so
  honestly; never use outside knowledge; cite every claim with [n] mapping to a
  chunk's source + page.
- Wrap each retrieved chunk as: "[n] (source: <file>, p.<page>) <text>".
- Place the <context> block in the latest USER turn (keeps the system prompt
  stable / cache-friendly and ties context to the question).
- Append trimmed recent history (settings.max_history_turns) for multi-turn,
  re-retrieving per turn rather than re-injecting old chunks.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant. Answer ONLY using the text inside the "
    "<context> block. If the answer is not in the context, say you don't have "
    "that information in the knowledge base — never use outside knowledge. Cite "
    "every factual claim with the matching [n] marker."
)


def build_messages(question: str, chunks, history=None):
    """Build the Anthropic Messages payload (system + history + grounded user turn).

    TODO: implement (Phase 4).
    """
    raise NotImplementedError("prompt_builder.build_messages — Phase 4")
