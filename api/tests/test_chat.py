"""Chat tests — prompt builder, no-context guard, SSE event shape, route validation.

The Claude token producer is monkeypatched so these run without an API key or
network. Async generators are driven via asyncio.run.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import chromadb
from fastapi.testclient import TestClient

from embeddings import embedder
from ingestion.chunker import Chunk
from llm import claude_stream, prompt_builder
from main import app
from routes import chat as chat_route
from vectorstore import chroma_store

client = TestClient(app)


def _collect(agen):
    async def run():
        return [e async for e in agen]

    return asyncio.run(run())


def _events(raw):
    """Turn SSE dicts into (event, parsed_data) tuples."""
    return [(e["event"], json.loads(e["data"])) for e in raw]


# ---------------------------------------------------------------- prompt builder


def test_build_messages_structure_and_citations():
    chunks = [
        {
            "document": "Refunds within 14 days.",
            "file_name": "policy.pdf",
            "start_page": 4,
            "end_page": 4,
            "chunk_index": 11,
        },
        {
            "document": "HQ in Bangalore.",
            "file_name": "about.pdf",
            "start_page": 2,
            "end_page": 3,
            "chunk_index": 0,
        },
    ]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    system, messages, citations = prompt_builder.build_messages("refund window?", chunks, history)

    assert system == prompt_builder.SYSTEM_PROMPT
    # history preserved, then a final grounded user turn
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1] == {"role": "assistant", "content": "hello"}
    last = messages[-1]
    assert last["role"] == "user"
    assert "<context>" in last["content"] and "[1]" in last["content"] and "[2]" in last["content"]
    assert "refund window?" in last["content"]
    # citations map 1:1 to markers
    assert [c["id"] for c in citations] == [1, 2]
    assert citations[0]["file_name"] == "policy.pdf" and citations[0]["page"] == 4


def test_no_context_message_is_embedded_in_system_prompt():
    # The refusal string must be the exact constant streamed on the no-context path.
    assert prompt_builder.NO_CONTEXT_MESSAGE in prompt_builder.SYSTEM_PROMPT


# ---------------------------------------------------------------- streaming


def test_stream_no_context_path():
    events = _events(_collect(claude_stream.stream_answer("", "anything", chunks=[], history=[])))
    kinds = [e for e, _ in events]
    assert kinds == ["token", "citations", "done"]
    assert events[0][1]["text"] == prompt_builder.NO_CONTEXT_MESSAGE
    assert events[1][1] == []  # empty citations
    assert events[2][1]["no_context"] is True


def test_stream_with_context_mocked(monkeypatch):
    async def fake_tokens(api_key, model, system, messages, meta=None):
        for t in ["Refunds ", "are issued within 14 days [1]."]:
            yield t

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", fake_tokens)

    chunk = {
        "document": "Refunds within 14 days.",
        "file_name": "policy.pdf",
        "start_page": 4,
        "end_page": 4,
        "chunk_index": 11,
    }
    events = _events(
        _collect(claude_stream.stream_answer("", "refund?", [chunk], [], api_key="test-key"))
    )

    kinds = [e for e, _ in events]
    assert kinds[-2:] == ["citations", "done"]
    tokens = "".join(d["text"] for k, d in events if k == "token")
    assert tokens == "Refunds are issued within 14 days [1]."
    citations = next(d for k, d in events if k == "citations")
    assert citations[0]["file_name"] == "policy.pdf" and citations[0]["id"] == 1
    assert events[-1][1]["no_context"] is False


def test_stream_missing_api_key_errors(monkeypatch):
    # Null out the env fallback so the empty key actually triggers the error path
    # (otherwise it falls back to settings.anthropic_api_key from the container env).
    monkeypatch.setattr(claude_stream.settings, "anthropic_api_key", "")
    chunk = {
        "document": "x",
        "file_name": "f.pdf",
        "start_page": 1,
        "end_page": 1,
        "chunk_index": 0,
    }
    events = _events(_collect(claude_stream.stream_answer("", "q", [chunk], [], api_key=None)))
    assert events[-1][0] == "error"


# ---------------------------------------------------------------- route validation


def test_chat_empty_message_422():
    assert client.post("/chat", json={"session_id": "s", "message": "   "}).status_code == 422


def test_chat_empty_session_422():
    assert client.post("/chat", json={"session_id": "  ", "message": "hi"}).status_code == 422


# ---------------------------------------------------------------- history sanitizer


def test_build_messages_sanitizes_history():
    # Starts with a stray assistant turn and ends with a dangling user turn (from an
    # interrupted/failed prior turn) — both invalid to replay to the Messages API.
    history = [
        {"role": "assistant", "content": "stray leading assistant"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "dangling user with no reply"},
    ]
    chunk = {
        "document": "x",
        "file_name": "f.pdf",
        "start_page": 1,
        "end_page": 1,
        "chunk_index": 0,
    }
    _, messages, _ = prompt_builder.build_messages("current?", [chunk], history)

    roles = [m["role"] for m in messages]
    assert roles[0] == "user"  # never starts with assistant
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))  # alternating
    assert messages[-1]["role"] == "user" and "current?" in messages[-1]["content"]


# ------------------------------------------------- retrieval / no-context guard


def _seed_one(text="Refunds are issued within 14 days of delivery."):
    coll = chroma_store.get_collection(
        client=chromadb.EphemeralClient(), name=f"t_{uuid.uuid4().hex}"
    )
    ch = Chunk(text=text, chunk_index=0, start_page=1, end_page=1)
    chroma_store.add_chunks(coll, "F", "f.pdf", [ch], embedder.embed_texts([text]))
    return coll


def test_retrieve_empty_collection(monkeypatch):
    empty = chroma_store.get_collection(
        client=chromadb.EphemeralClient(), name=f"t_{uuid.uuid4().hex}"
    )
    monkeypatch.setattr(chat_route.chroma_store, "get_collection", lambda *a, **k: empty)
    assert chat_route._retrieve("anything", 5) == []


def test_retrieve_relevant_passes_threshold(monkeypatch):
    coll = _seed_one()
    monkeypatch.setattr(chat_route.chroma_store, "get_collection", lambda *a, **k: coll)
    hits = chat_route._retrieve("what is the refund window?", 5)
    assert hits and hits[0]["score"] >= chat_route.settings.relevance_threshold


def test_retrieve_below_threshold_returns_empty(monkeypatch):
    coll = _seed_one()
    monkeypatch.setattr(chat_route.chroma_store, "get_collection", lambda *a, **k: coll)
    monkeypatch.setattr(chat_route.settings, "relevance_threshold", 0.99)
    assert chat_route._retrieve("what is the refund window?", 5) == []


# ----------------------------------------------- Phase-6 hardening regressions

_CHUNK = {
    "document": "Refunds within 14 days.",
    "file_name": "policy.pdf",
    "start_page": 4,
    "end_page": 4,
    "chunk_index": 11,
}


def test_stream_midstream_error_still_emits_citations_and_done(monkeypatch):
    async def boom(api_key, model, system, messages, meta=None):
        yield "Partial answer [1]"
        raise RuntimeError("connection dropped")

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", boom)
    events = _events(_collect(claude_stream.stream_answer("", "q", [_CHUNK], [], api_key="k")))
    kinds = [e for e, _ in events]
    assert "citations" in kinds and kinds[-1] == "done"  # citations survive a mid-stream error
    assert any(k == "error" for k in kinds)
    assert events[-1][1].get("interrupted") is True


def test_stream_empty_completion_falls_back(monkeypatch):
    async def empty(api_key, model, system, messages, meta=None):
        return
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(claude_stream, "_stream_claude_tokens", empty)
    events = _events(_collect(claude_stream.stream_answer("", "q", [_CHUNK], [], api_key="k")))
    # No empty bubble with sources: a fallback token + EMPTY citations.
    assert events[0][0] == "token" and events[0][1]["text"] == claude_stream.EMPTY_ANSWER_MESSAGE
    citations = next(d for k, d in events if k == "citations")
    assert citations == []


def test_build_messages_trims_to_token_budget(monkeypatch):
    monkeypatch.setattr(prompt_builder.settings, "max_context_tokens", 200)
    big = "word " * 400  # ~2000 chars -> ~500 approx tokens, over the 200 budget
    history = []
    for i in range(6):
        history.append({"role": "user", "content": f"q{i} {big}"})
        history.append({"role": "assistant", "content": f"a{i} {big}"})
    _, messages, _ = prompt_builder.build_messages("current question?", [_CHUNK], history)
    # Trimmed (not all 12 history turns survive), still valid + current turn kept.
    assert len(messages) < 13
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user" and "current question?" in messages[-1]["content"]


def test_chat_retrieval_failure_streams_error_not_500(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(chat_route, "_retrieve", boom)
    r = client.post("/chat", json={"session_id": "err1", "message": "hello"})
    assert r.status_code == 200  # graceful SSE, not a 500
    assert "Could not search the knowledge base" in r.text
