"""PDF text extraction (PyMuPDF primary, pdfplumber for table-heavy pages).

- PyMuPDF block-sorted text reconstructs reading order for multi-column layouts.
- Pages with detected tables get a pdfplumber rendering appended.
- Scanned / image-only pages yield empty text so the caller can flag the file as
  `no_extractable_text` and continue the batch.
- Corrupt / password-protected PDFs raise ValueError so the caller records a
  per-file error instead of aborting the whole sync.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A page with fewer than this many extracted chars is treated as having no usable
# text (scanned/image-only) for flagging purposes.
_MIN_CHARS_PER_PAGE = 1


@dataclass
class PageText:
    """Extracted text for a single PDF page (1-indexed)."""

    page_number: int
    text: str


def _extract_page_blocks(page) -> str:
    """Extract a page's text in reading order using block coordinates.

    PyMuPDF "blocks" yields (x0, y0, x1, y1, text, block_no, block_type). Sorting
    by (y0, x0) reconstructs reading order for multi-column pages better than the
    raw stream order.
    """
    blocks = page.get_text("blocks")
    text_blocks = [
        b for b in blocks
        if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()
    ]
    text_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return "\n".join(b[4].strip() for b in text_blocks).strip()


def _extract_tables_pdfplumber(pdf_bytes: bytes, page_index: int) -> str:
    """Render tables on a 0-indexed page to pipe-delimited text via pdfplumber."""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_index >= len(pdf.pages):
                return ""
            tables = pdf.pages[page_index].extract_tables() or []
            rendered: list[str] = []
            for table in tables:
                for row in table:
                    cells = [(c or "").strip().replace("\n", " ") for c in row]
                    if any(cells):
                        rendered.append(" | ".join(cells))
            return "\n".join(rendered).strip()
    except Exception as exc:  # pragma: no cover - tables are best-effort
        logger.warning("pdfplumber failed on page %d: %s", page_index + 1, exc)
        return ""


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Extract per-page text from PDF bytes, preserving reading order."""
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    if getattr(doc, "needs_pass", False):
        doc.close()
        raise ValueError("PDF is password-protected")

    pages: list[PageText] = []
    try:
        for i, page in enumerate(doc):
            try:
                text = _extract_page_blocks(page)
            except Exception as exc:  # pragma: no cover - per-page robustness
                logger.warning("PyMuPDF extraction failed on page %d: %s", i + 1, exc)
                text = ""
            # Append tables (best-effort) when the page appears to contain them.
            try:
                if page.find_tables().tables:
                    table_text = _extract_tables_pdfplumber(pdf_bytes, i)
                    if table_text:
                        text = f"{text}\n{table_text}".strip() if text else table_text
            except Exception:
                pass
            pages.append(PageText(page_number=i + 1, text=text))
    finally:
        doc.close()
    return pages


def has_extractable_text(pages: list[PageText]) -> bool:
    """True if any page yielded usable text (else likely scanned/image-only)."""
    return any(len(p.text.strip()) >= _MIN_CHARS_PER_PAGE for p in pages)
