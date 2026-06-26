---
name: chunk-inspect
description: Show how a given PDF chunks under the current settings — chunk count, per-chunk page spans, and token sizes (counted with the embedding model's tokenizer), plus summary stats. Use to validate the chunker on tricky PDFs (multi-column, tables, scanned) and to justify the chunking strategy in the README (15-pt parsing/chunking criterion).
---

# chunk-inspect — inspect a PDF's chunking

Drives `api/scripts/chunk_inspect.py`, which runs the real `pdf_parser` + `chunker`
+ the embedding model's tokenizer, so the numbers match production.

## Steps
1. Make sure the api container is up: `docker compose ps` (start with `docker compose up -d` if not).
2. Pick a PDF. It can be any host path (e.g. one of `docs/*.pdf`, or a sample from the corpus).
3. Copy it into the container and run the inspector:
   ```bash
   docker compose cp "<path/to/file.pdf>" api:/tmp/inspect.pdf
   # Git Bash/Windows: prefix with MSYS_NO_PATHCONV=1 so /tmp isn't path-mangled.
   MSYS_NO_PATHCONV=1 docker compose exec -T api python scripts/chunk_inspect.py /tmp/inspect.pdf
   ```
   (PowerShell doesn't need the `MSYS_NO_PATHCONV=1` prefix.)
4. Read the report:
   - **pages / extractable_text** — confirms parsing worked (scanned PDFs show `extractable_text: False`).
   - **effective_chunk_tokens / max_seq / overlap** — the active sizing (clamped to the model's max).
   - **chunks + token sizes** (min/max/mean/median) — confirm none exceed the target and sizes are sensible.
   - **per-chunk page spans** — verify cross-page chunks carry both `start_page` and `end_page`.

## Use it to
- Sanity-check a multi-column or table-heavy PDF extracts in reading order.
- Tune `chunk_tokens` / `chunk_overlap` in `api/config.py` and re-run to compare.
- Capture concrete numbers for the README's chunking justification (in your own words).

## Notes
- First run loads the embedding model (cached on the app-data volume).
- Read-only: it never touches Chroma or the SQLite store.
