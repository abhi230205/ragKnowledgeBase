"""Sentence-aware sliding-window chunker with overlap, page-tracked.

Overlap is non-negotiable (the brief penalises "no chunk overlap" and tests
questions that span paragraph breaks).

TODO (Phase 2):
- ~800-token windows with ~120-token (~15%) overlap, snapping boundaries to
  sentence ends so chunks don't cut mid-sentence.
- Chunk ACROSS page boundaries (don't lose cross-page context); record
  start_page/end_page per chunk for citations.
- Edge cases: input shorter than one window -> exactly one chunk; empty/
  whitespace-only input -> zero chunks (no crash).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of text with page attribution and a stable index."""

    text: str
    chunk_index: int
    start_page: int
    end_page: int


def chunk_pages(
    pages,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Split extracted pages into overlapping, page-tracked chunks.

    TODO: implement (Phase 2). `pages` is a list of pdf_parser.PageText.
    """
    raise NotImplementedError("chunker.chunk_pages — Phase 2")
