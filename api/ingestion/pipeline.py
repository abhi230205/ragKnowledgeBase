"""The ingestion run: Drive -> diff -> (download -> parse -> chunk -> embed -> upsert).

Orchestrates the modular pieces (drive_client, pdf_parser, chunker, embedder,
chroma_store, crud). Per-file try/except means one bad file is recorded in
`errors[]` and the batch continues; state is written per file so a re-run resumes
cleanly.

Credentials/folder come from the SQLite config (UI-entered, Phase 5) or fall back
to env settings — never hardcoded.
"""

from __future__ import annotations

import logging

from config import settings
from db import crud
from db.session import get_session
from embeddings import embedder
from vectorstore import chroma_store

from ingestion import chunker, docx_parser, pdf_parser
from ingestion.drive_client import (
    DOCX_MIME,
    PDF_MIME,
    DriveAuthError,
    DriveClient,
    DriveError,
    extract_folder_id,
)
from ingestion.sync_diff import compute_diff

logger = logging.getLogger(__name__)


def run_sync(folder_id: str | None = None, progress_cb=None) -> dict:
    """Run one ingestion pass. Returns a status summary. Normal operational
    failures (auth/folder/listing) come back in the dict rather than raising.

    `progress_cb(done, total, file_name, phase)` is called as each file is
    processed, so callers can stream a live progress bar.
    """
    session = get_session()
    try:
        cfg = crud.get_or_create_config(session)
        folder = folder_id or cfg.drive_folder_id or settings.drive_folder_id
        if not folder:
            return {"status": "error", "error": "No Drive folder id configured."}
        folder = extract_folder_id(folder)  # tolerate a pasted folder URL

        info = crud.get_service_account_info(cfg)
        path = settings.google_service_account_path
        try:
            client = DriveClient(service_account_path=path, service_account_info=info)
        except DriveAuthError as exc:
            logger.warning("Drive auth failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        try:
            drive_files = client.list_documents(folder)
        except (DriveError, DriveAuthError) as exc:
            logger.warning("Drive listing failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        # Embedding-model guard: if the active model/dimension differs from what the
        # collection was indexed with, reset it and re-embed everything (instead of a
        # silent per-file dimension-mismatch error storm). Self-heals an env change.
        reindexed = False
        try:
            reindexed = chroma_store.ensure_model(
                chroma_store.get_client(), settings.embedding_model, embedder.dimension()
            )
        except Exception:  # pragma: no cover - guard must never break a sync
            logger.exception("Embedding-model guard failed (continuing)")
        if reindexed:
            logger.warning("Embedding model changed -> reset collection, re-indexing all files")
            for r in crud.list_files(session):
                crud.delete_file(session, r.file_id)

        # Snapshot existing chunk counts once (for chunks_deleted accounting).
        prior_counts = {r.file_id: (r.chunk_count or 0) for r in crud.list_files(session)}
        tracked = crud.get_tracked(session)
        diff = compute_diff(drive_files, tracked)
        collection = chroma_store.get_collection()

        counts = {
            "files_total": len(drive_files),
            "reindexed": reindexed,
            "added": 0,
            "modified": 0,
            "deleted": 0,
            "renamed": 0,
            "unchanged": len(diff.unchanged),
            "chunks_created": 0,
            "chunks_deleted": 0,
            "errors": [],
        }

        # Progress accounting: total = files we act on this run (skip "unchanged").
        total = len(diff.deleted) + len(diff.renamed) + len(diff.modified) + len(diff.added)
        done = 0

        def _tick(name: str | None, phase: str) -> None:
            nonlocal done
            done += 1
            if progress_cb:
                progress_cb(done, total, name, phase)

        if progress_cb:
            progress_cb(0, total, None, "starting")

        # --- deleted: drop chunks + record row ---
        for file_id in diff.deleted:
            try:
                chroma_store.delete_file(collection, file_id)
                crud.delete_file(session, file_id)
                counts["deleted"] += 1
                counts["chunks_deleted"] += prior_counts.get(file_id, 0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed deleting %s", file_id)
                counts["errors"].append({"file": file_id, "reason": f"delete failed: {exc}"})
            _tick(file_id, "deleting")

        # --- renamed: metadata-only update, no re-embedding ---
        for f in diff.renamed:
            try:
                chroma_store.update_file_name(collection, f.id, f.name)
                crud.rename_file(session, f.id, f.name)
                counts["renamed"] += 1
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed renaming %s", f.id)
                counts["errors"].append({"file": f.name, "reason": f"rename failed: {exc}"})
            _tick(f.name, "renaming")

        # --- modified: delete old chunks first (no orphans), then re-process ---
        for f in diff.modified:
            try:
                # Mark the row transitional (keep the OLD md5, 0 chunks, status=pending)
                # BEFORE deleting chunks, so a crash between delete and re-embed leaves
                # an honest, resumable state (next sync still re-detects 'modified')
                # rather than "embedded" with chunks that no longer exist.
                old_md5 = (tracked.get(f.id) or {}).get("md5_checksum")
                crud.upsert_file(
                    session, f.id, f.name, old_md5, f.modified_time, 0, status="pending"
                )
                chroma_store.delete_file(collection, f.id)
                counts["chunks_deleted"] += prior_counts.get(f.id, 0)
                if _process_file(session, collection, client, f, counts) is not None:
                    counts["modified"] += 1
            except Exception as exc:
                logger.exception("Failed re-processing %s", f.id)
                crud.set_file_error(session, f.id, f.name, str(exc))
                counts["errors"].append({"file": f.name, "reason": str(exc)})
            _tick(f.name, "embedding")

        # --- added: process from scratch ---
        for f in diff.added:
            try:
                if _process_file(session, collection, client, f, counts) is not None:
                    counts["added"] += 1
            except Exception as exc:
                logger.exception("Failed processing %s", f.id)
                crud.set_file_error(session, f.id, f.name, str(exc))
                counts["errors"].append({"file": f.name, "reason": str(exc)})
            _tick(f.name, "embedding")

        summary = crud.status_summary(session)
        return {
            "status": "completed",
            "summary": counts,
            "documents": summary["documents"],
            "chunks": summary["chunks"],
            "last_sync": summary["last_sync"],
        }
    finally:
        session.close()


def _process_file(session, collection, client: DriveClient, f, counts) -> int | None:
    """Download → parse → chunk → embed → upsert one file.

    Returns the chunk count, or None if the file was flagged (e.g. no extractable
    text). Raises on hard errors so the caller records a per-file error.
    """
    raw = client.download_bytes(f.id)

    # Dispatch by type: DOCX via python-docx, everything else via PyMuPDF. Trust the
    # authoritative Drive mime type; only fall back to the filename extension when the
    # mime is neither known type (so a PDF mislabeled "*.docx" still parses as a PDF).
    if f.mime_type == DOCX_MIME:
        is_docx = True
    elif f.mime_type == PDF_MIME:
        is_docx = False
    else:
        is_docx = f.name.lower().endswith(".docx")
    parser = docx_parser if is_docx else pdf_parser

    try:
        pages = parser.extract_pages(raw)
    except ValueError as exc:
        crud.set_file_error(session, f.id, f.name, str(exc))
        counts["errors"].append({"file": f.name, "reason": str(exc)})
        return None

    if not pdf_parser.has_extractable_text(pages):
        crud.upsert_file(
            session,
            f.id,
            f.name,
            f.md5_checksum,
            f.modified_time,
            0,
            status="no_extractable_text",
        )
        counts["errors"].append({"file": f.name, "reason": "no_extractable_text"})
        return None

    chunks = chunker.chunk_pages(
        pages,
        chunk_tokens=embedder.effective_chunk_tokens(),
        chunk_overlap=settings.chunk_overlap,
        token_counter=embedder.token_counter(),
    )
    if not chunks:
        crud.upsert_file(
            session,
            f.id,
            f.name,
            f.md5_checksum,
            f.modified_time,
            0,
            status="no_extractable_text",
        )
        counts["errors"].append({"file": f.name, "reason": "no_extractable_text"})
        return None

    # Embed + upsert in fixed-size slices so peak memory is bounded by the batch,
    # not by the document size (very large PDF edge case).
    batch = settings.embed_batch_chunks
    for i in range(0, len(chunks), batch):
        sl = chunks[i : i + batch]
        vecs = embedder.embed_texts([c.text for c in sl])
        chroma_store.add_chunks(collection, f.id, f.name, sl, vecs)
        del vecs

    crud.upsert_file(
        session,
        f.id,
        f.name,
        f.md5_checksum,
        f.modified_time,
        len(chunks),
        status="embedded",
    )
    counts["chunks_created"] += len(chunks)
    return len(chunks)
