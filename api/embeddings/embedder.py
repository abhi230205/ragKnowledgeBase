"""Local embedding model wrapper (sentence-transformers).

Default model: all-MiniLM-L6-v2 (384-dim). INVARIANT: the same model must be used
for indexing and querying, and its dimension must match the Chroma collection —
switching models requires a full re-index (Phase 2/8 edge case).

Claude is NOT an embedding model; embeddings are computed locally and never via
the Anthropic API.

TODO (Phase 2):
- Lazy-load the model once (weights cached on the app-data volume via HF_HOME /
  SENTENCE_TRANSFORMERS_HOME) so the image build and /health stay fast.
- embed_texts(list[str]) -> list[list[float]]   (batched)
- embed_query(str) -> list[float]
- Run the CPU-bound encode off the event loop (run_in_threadpool) when called
  from async routes.
- dimension() -> int for validating the Chroma collection.
"""

from __future__ import annotations


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts. TODO: implement (Phase 2)."""
    raise NotImplementedError("embedder.embed_texts — Phase 2")


def embed_query(text: str) -> list[float]:
    """Embed a single query string. TODO: implement (Phase 2)."""
    raise NotImplementedError("embedder.embed_query — Phase 2")
