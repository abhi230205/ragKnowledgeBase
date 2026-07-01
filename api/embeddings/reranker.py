"""Cross-encoder re-ranking layer (bonus).

A bi-encoder (the embedder) retrieves fast by cosine similarity but scores the
query and each chunk independently. A cross-encoder scores the (query, chunk)
PAIR jointly, so it judges relevance far more accurately — at higher cost. So the
pipeline is: retrieve the top-N candidates cheaply by cosine, then re-score just
those N with the cross-encoder and keep the best top-k.

Local model (sentence-transformers CrossEncoder, default ms-marco-MiniLM) — no API
call, cached on the app-data volume like the embedder. Toggled by
settings.rerank_enabled; when off, retrieval is pure cosine top-k (unchanged).
"""

from __future__ import annotations

import logging
import threading

from config import settings

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None
_lock = threading.Lock()


def is_enabled() -> bool:
    return bool(settings.rerank_enabled)


def candidate_count(top_k: int) -> int:
    """How many candidates to pull from the vector store before re-ranking.

    When re-ranking is off this is just top_k (behaviour identical to before).
    """
    return max(top_k, settings.rerank_candidates) if is_enabled() else top_k


def get_model():
    """Lazily load and cache the CrossEncoder (thread-safe)."""
    global _model, _model_name
    if _model is not None and _model_name == settings.rerank_model:
        return _model
    with _lock:
        if _model is None or _model_name != settings.rerank_model:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", settings.rerank_model)
            _model = CrossEncoder(settings.rerank_model)
            _model_name = settings.rerank_model
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-order candidate hit dicts by cross-encoder relevance; return the top_k.

    `candidates` are chroma_store.query() hits (each with a 'document' text). Each
    returned hit gains a 'rerank_score' and a fresh 1-based 'rank'. If re-ranking
    is disabled (or there's nothing to reorder) this just truncates to top_k,
    preserving the original cosine order — a no-op relative to the old behaviour.
    CPU-bound: callers already run it off the event loop (threadpool / bg thread).
    """
    if not candidates:
        return []
    if not is_enabled() or len(candidates) <= 1:
        out = candidates[:top_k]
    else:
        model = get_model()
        pairs = [(query, c.get("document") or c.get("preview") or "") for c in candidates]
        scores = model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = round(float(score), 4)
        out = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
    for i, c in enumerate(out, start=1):
        c["rank"] = i
    return out
