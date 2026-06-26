"""Sentence-aware sliding-window chunker with overlap, page-tracked.

Chunks are measured in *tokens* via an injected counter so sizes line up with
what the embedding model actually encodes (all-MiniLM-L6-v2 truncates at 256
tokens — see embeddings.embedder.effective_chunk_tokens). Overlap is mandatory
(the brief penalises its absence and tests questions that span boundaries).

Design:
- Split each page into sentences (sentence enders + newlines); a sentence longer
  than the window is sub-split by words so no piece exceeds the window.
- Greedily pack sentences into windows up to `chunk_tokens`.
- Start each next window with the trailing sentences of the previous one whose
  token sum fits in `chunk_overlap` (so a sentence straddling a boundary appears
  whole in at least one chunk).
- Chunk across page boundaries; record start_page/end_page for citations.

Edge cases: empty/whitespace input → zero chunks; input shorter than one window
→ exactly one chunk; the index math always advances (no infinite loop).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


# Default counter (word-based) — fast and dependency-free, used in tests. The
# pipeline injects the embedding model's tokenizer for production accuracy.
def default_token_counter(text: str) -> int:
    return len(text.split())


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Chunk:
    """A single chunk of text with page attribution and a stable index."""

    text: str
    chunk_index: int
    start_page: int
    end_page: int


@dataclass
class _Sent:
    text: str
    page: int
    ntok: int


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def _split_long_sentence(
    sentence: str, max_tokens: int, token_counter: Callable[[str], int]
) -> list[str]:
    """Split a sentence that exceeds the window into <=max_tokens word-pieces."""
    words = sentence.split()
    pieces: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if token_counter(" ".join(cur)) > max_tokens:
            cur.pop()
            if cur:
                pieces.append(" ".join(cur))
            cur = [w]
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [sentence]


def _build_sentences(pages, chunk_tokens: int, token_counter: Callable[[str], int]) -> list[_Sent]:
    sentences: list[_Sent] = []
    for page in pages:
        for raw in _split_sentences(page.text or ""):
            ntok = token_counter(raw)
            if ntok > chunk_tokens:
                for piece in _split_long_sentence(raw, chunk_tokens, token_counter):
                    sentences.append(_Sent(piece, page.page_number, token_counter(piece)))
            else:
                sentences.append(_Sent(raw, page.page_number, ntok))
    return sentences


def chunk_pages(
    pages,
    chunk_tokens: int = 256,
    chunk_overlap: int = 38,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Split extracted pages into overlapping, page-tracked chunks.

    `pages` is a list of pdf_parser.PageText. `token_counter` defaults to a word
    count; the pipeline passes the embedding model's tokenizer.
    """
    counter = token_counter or default_token_counter
    chunk_overlap = max(0, min(chunk_overlap, chunk_tokens - 1))

    sentences = _build_sentences(pages, chunk_tokens, counter)
    n = len(sentences)
    chunks: list[Chunk] = []
    i = 0
    while i < n:
        # Grow a window from i until adding the next sentence would overflow.
        j = i
        cur_tokens = 0
        while j < n:
            ntok = sentences[j].ntok
            if cur_tokens + ntok > chunk_tokens and j > i:
                break
            cur_tokens += ntok
            j += 1

        window = sentences[i:j]
        pages_in = [s.page for s in window]
        chunks.append(
            Chunk(
                text=" ".join(s.text for s in window),
                chunk_index=len(chunks),
                start_page=min(pages_in),
                end_page=max(pages_in),
            )
        )

        if j >= n:
            break

        # Compute overlap: trailing sentences whose token sum fits the budget.
        ov_tokens = 0
        k = j
        while k > i:
            if ov_tokens + sentences[k - 1].ntok > chunk_overlap:
                break
            ov_tokens += sentences[k - 1].ntok
            k -= 1
        # Always advance (k == i would mean the whole window fit in the overlap
        # budget — then start the next window after it).
        i = k if k > i else j

    return chunks
