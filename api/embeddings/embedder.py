"""Local embedding model wrapper (sentence-transformers).

Default model: all-MiniLM-L6-v2 (384-dim, max 256 tokens). INVARIANT: the same
model is used for indexing and querying, and its dimension must match the Chroma
collection — switching models requires a full re-index (handled in chroma_store
+ a settings warning later).

Claude is NOT an embedding model; embeddings are computed locally, never via the
Anthropic API. The model is loaded once (lazily) and cached; weights are cached
on the app-data volume via HF_HOME / SENTENCE_TRANSFORMERS_HOME so the image
build and /health stay fast.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from config import settings

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None
_lock = threading.Lock()


def get_model():
    """Lazily load and cache the SentenceTransformer model (thread-safe)."""
    global _model, _model_name
    if _model is not None and _model_name == settings.embedding_model:
        return _model
    with _lock:
        if _model is None or _model_name != settings.embedding_model:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
            _model_name = settings.embedding_model
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of texts → list of unit-normalised vectors.

    CPU-bound; call via run_in_threadpool/asyncio.to_thread from async code. The
    ingestion pipeline already runs in a background (APScheduler) thread.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


def dimension() -> int:
    """Embedding vector dimension (must match the Chroma collection)."""
    model = get_model()
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def max_content_tokens() -> int:
    """Max input tokens the model encodes before truncation (e.g. 256 for MiniLM)."""
    return int(getattr(get_model(), "max_seq_length", 256) or 256)


def count_tokens(text: str) -> int:
    """Count tokens with the model's own tokenizer (no special tokens)."""
    return len(get_model().tokenizer.encode(text, add_special_tokens=False))


def token_counter() -> Callable[[str], int]:
    """Return a token-counting callable bound to the model tokenizer.

    Injected into the chunker so chunk sizes are measured in the same tokens the
    model truncates on. Tests can inject a simpler counter instead.
    """
    model = get_model()

    def _count(text: str) -> int:
        return len(model.tokenizer.encode(text, add_special_tokens=False))

    return _count


def effective_chunk_tokens() -> int:
    """Chunk-size target clamped to the model's max (minus special-token margin)."""
    return min(settings.chunk_tokens, max_content_tokens() - settings.chunk_token_margin)
