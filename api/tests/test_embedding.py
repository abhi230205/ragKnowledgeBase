"""Embedding tests — dimension, determinism, batch length.

These load the sentence-transformers model (downloaded once, cached on the
app-data volume), so run them in the container: `docker compose exec api pytest`.
"""

from __future__ import annotations

import pytest

from embeddings import embedder


def test_embedding_dimension_is_384():
    assert embedder.dimension() == 384


def test_embed_texts_batch_length_and_dim():
    texts = ["hello world", "the refund policy is 14 days", "bangalore india"]
    vectors = embedder.embed_texts(texts)
    assert len(vectors) == len(texts)
    assert all(len(v) == 384 for v in vectors)


def test_embedding_is_deterministic():
    a = embedder.embed_query("how long is the return window?")
    b = embedder.embed_query("how long is the return window?")
    assert a == pytest.approx(b)


def test_empty_input_returns_empty():
    assert embedder.embed_texts([]) == []
