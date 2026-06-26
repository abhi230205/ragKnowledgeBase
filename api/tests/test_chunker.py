"""Chunker tests — count, overlap, max size, page metadata, short/empty docs.

These use the chunker's default word-based token counter, so they're fast and
need no model download.
"""

from __future__ import annotations

from ingestion.chunker import Chunk, chunk_pages, default_token_counter
from ingestion.pdf_parser import PageText


def _page(n: int, text: str) -> PageText:
    return PageText(page_number=n, text=text)


def test_chunk_count_no_overlap():
    # 10 sentences x 10 tokens, window 25, overlap 0 -> 2 sentences/chunk -> 5 chunks.
    sentences = " ".join(" ".join(f"w{i}" for _ in range(10)) + "." for i in range(10))
    chunks = chunk_pages([_page(1, sentences)], chunk_tokens=25, chunk_overlap=0)
    assert len(chunks) == 5


def test_chunk_overlap_shares_text():
    # Unique sentences; window holds 2, overlap carries the last one forward.
    text = " ".join(f"alpha{i} beta gamma delta epsilon." for i in range(8))
    chunks = chunk_pages([_page(1, text)], chunk_tokens=12, chunk_overlap=5)
    assert len(chunks) >= 2
    # Consecutive chunks must share a sentence marker (overlap present).
    assert "alpha1" in chunks[0].text and "alpha1" in chunks[1].text


def test_chunk_never_exceeds_max():
    text = " ".join(f"alpha{i} beta gamma delta epsilon." for i in range(20))
    chunks = chunk_pages([_page(1, text)], chunk_tokens=12, chunk_overlap=4)
    for c in chunks:
        assert default_token_counter(c.text) <= 12


def test_chunk_page_metadata_spans_pages():
    # window 15 lets a chunk straddle the page-1/page-2 boundary.
    p1 = _page(1, "a b c d e. f g h i j.")
    p2 = _page(2, "k l m n o. p q r s t.")
    chunks = chunk_pages([p1, p2], chunk_tokens=15, chunk_overlap=0)
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 2
    # Indices are sequential and start at 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_short_doc_single_chunk():
    chunks = chunk_pages([_page(1, "one two three.")], chunk_tokens=256, chunk_overlap=38)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].start_page == 1 and chunks[0].end_page == 1


def test_empty_input_no_chunks():
    assert chunk_pages([], chunk_tokens=256) == []
    assert chunk_pages([_page(1, "   \n  ")], chunk_tokens=256) == []
