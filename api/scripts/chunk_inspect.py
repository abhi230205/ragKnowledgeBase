"""Chunk-inspection utility: show how a PDF chunks (counts, page spans, token sizes).

Runs the real pipeline pieces (pdf_parser + chunker + the embedding model's
tokenizer) so the numbers match production. Useful for documenting/justifying the
chunking strategy in the README (the 15-pt parsing/chunking criterion).

Usage (inside the api container, which has the deps + model):
    docker compose cp <file.pdf> rag-api:/tmp/inspect.pdf
    docker compose exec -T api python scripts/chunk_inspect.py /tmp/inspect.pdf
"""

from __future__ import annotations

import os
import statistics
import sys

# Ensure /app (the package root) is importable regardless of how we're invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from embeddings import embedder  # noqa: E402
from ingestion import chunker, pdf_parser  # noqa: E402


def main(path: str) -> int:
    with open(path, "rb") as fh:
        data = fh.read()

    try:
        pages = pdf_parser.extract_pages(data)
    except ValueError as exc:
        print(f"Could not parse PDF: {exc}")
        return 1

    print(f"PDF: {path}")
    print(f"pages: {len(pages)}  extractable_text: {pdf_parser.has_extractable_text(pages)}")
    if not pdf_parser.has_extractable_text(pages):
        print("No extractable text (scanned / image-only?). Nothing to chunk.")
        return 0

    chunk_tokens = embedder.effective_chunk_tokens()
    overlap = settings.chunk_overlap
    counter = embedder.token_counter()
    chunks = chunker.chunk_pages(
        pages, chunk_tokens=chunk_tokens, chunk_overlap=overlap, token_counter=counter
    )
    sizes = [counter(c.text) for c in chunks]

    print(
        f"model={settings.embedding_model}  max_seq={embedder.max_content_tokens()}  "
        f"effective_chunk_tokens={chunk_tokens}  overlap={overlap}"
    )
    print(f"chunks: {len(chunks)}")
    if sizes:
        print(
            f"token sizes -> min={min(sizes)} max={max(sizes)} "
            f"mean={statistics.mean(sizes):.1f} median={statistics.median(sizes)}"
        )
        print(f"chunks over target ({chunk_tokens}): {sum(1 for s in sizes if s > chunk_tokens)}")

    print()
    print(f"{'idx':>4} {'pages':>9} {'tokens':>6}  preview")
    for c, s in zip(chunks, sizes):
        span = str(c.start_page) if c.start_page == c.end_page else f"{c.start_page}-{c.end_page}"
        preview = c.text[:70].replace("\n", " ")
        print(f"{c.chunk_index:>4} {span:>9} {s:>6}  {preview}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/chunk_inspect.py <file.pdf>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
