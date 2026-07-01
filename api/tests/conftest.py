"""Shared pytest fixtures / safety guards for the suite."""

from __future__ import annotations

import pytest

from config import settings


@pytest.fixture(autouse=True, scope="session")
def _null_anthropic_key_for_tests():
    """Defense-in-depth: never let the test suite reach the live Anthropic API.

    The canonical command (`docker compose run --rm api pytest`) injects the real
    ANTHROPIC_API_KEY into the container env, so settings.anthropic_api_key is the
    live key during tests. Per-test fakes monkeypatch claude_stream._stream_claude_tokens
    to drive the streaming path, but if one were ever missing, a real (billed) call
    would fire. Nulling the key here makes that fail safely via the
    "Anthropic API key is not configured" path instead. Tests that need a key set it
    explicitly via monkeypatch.
    """
    original = settings.anthropic_api_key
    settings.anthropic_api_key = None
    # Gate the cross-encoder re-ranker OFF by default so retrieval tests stay fast
    # and deterministic (no ~80MB model download); test_reranker.py flips it on.
    original_rerank = settings.rerank_enabled
    settings.rerank_enabled = False
    yield
    settings.anthropic_api_key = original
    settings.rerank_enabled = original_rerank
