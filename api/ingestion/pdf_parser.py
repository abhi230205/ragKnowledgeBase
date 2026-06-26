"""PDF text extraction (PyMuPDF primary, pdfplumber for table-heavy pages).

TODO (Phase 2):
- Extract text per page in logical reading order (PyMuPDF block-level sorting
  reconstructs multi-column layouts).
- Route table-heavy pages through pdfplumber (better table extraction).
- Detect near-empty pages (scanned / image-only PDFs) and signal
  `no_extractable_text` so the caller can flag the file and continue the sync.
- Guard open/extract with try/except so a corrupt/password-protected PDF records
  a per-file error instead of aborting the whole batch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageText:
    """Extracted text for a single PDF page (1-indexed)."""

    page_number: int
    text: str


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Extract per-page text from PDF bytes. TODO: implement with PyMuPDF (Phase 2)."""
    raise NotImplementedError("pdf_parser.extract_pages — Phase 2")
