"""CRUD helpers over the config/files tables.

Used by the routes and the ingestion pipeline. Mutating file helpers commit
immediately so per-file sync progress persists (a crash mid-batch leaves a clean,
resumable state). Secret masking lives at the route layer, not here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Config, FileRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- config


def get_or_create_config(session: Session) -> Config:
    """Return the singleton config row (id=1), creating it if absent."""
    cfg = session.get(Config, 1)
    if cfg is None:
        cfg = Config(id=1)
        session.add(cfg)
        session.commit()
        session.refresh(cfg)
    return cfg


def update_config(session: Session, **fields) -> Config:
    """Update provided config fields (ignores None values)."""
    cfg = get_or_create_config(session)
    for key, value in fields.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    session.commit()
    session.refresh(cfg)
    return cfg


def get_service_account_info(cfg: Config) -> dict | None:
    """Parse the stored service-account JSON string into a dict, if present."""
    if not cfg.service_account_json:
        return None
    try:
        return json.loads(cfg.service_account_json)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------- files


def list_files(session: Session) -> list[FileRecord]:
    return list(session.scalars(select(FileRecord)).all())


def get_tracked(session: Session) -> dict[str, dict]:
    """Build the {file_id: {md5_checksum, file_name}} map for sync_diff."""
    return {
        r.file_id: {"md5_checksum": r.md5_checksum, "file_name": r.file_name}
        for r in list_files(session)
    }


def upsert_file(
    session: Session,
    file_id: str,
    file_name: str,
    md5_checksum: str | None,
    modified_time: str | None,
    chunk_count: int,
    status: str,
    error: str | None = None,
) -> FileRecord:
    """Insert or update a file record and stamp last_synced. Commits."""
    rec = session.get(FileRecord, file_id)
    if rec is None:
        rec = FileRecord(file_id=file_id)
        session.add(rec)
    rec.file_name = file_name
    rec.md5_checksum = md5_checksum
    rec.modified_time = modified_time
    rec.chunk_count = chunk_count
    rec.status = status
    rec.error = error
    rec.last_synced = _utcnow()
    session.commit()
    return rec


def rename_file(session: Session, file_id: str, new_name: str) -> None:
    """Update only the stored file name (rename without content change). Commits."""
    rec = session.get(FileRecord, file_id)
    if rec is not None:
        rec.file_name = new_name
        rec.last_synced = _utcnow()
        session.commit()


def set_file_error(session: Session, file_id: str, file_name: str, error: str) -> None:
    """Record a per-file error without aborting the batch. Commits."""
    rec = session.get(FileRecord, file_id)
    if rec is None:
        rec = FileRecord(file_id=file_id)
        session.add(rec)
    rec.file_name = file_name
    rec.status = "error"
    rec.error = error
    rec.last_synced = _utcnow()
    session.commit()


def delete_file(session: Session, file_id: str) -> None:
    """Remove a file record (after its chunks are deleted from Chroma). Commits."""
    rec = session.get(FileRecord, file_id)
    if rec is not None:
        session.delete(rec)
        session.commit()


def status_summary(session: Session) -> dict:
    """Aggregate counts for GET /status."""
    files = list_files(session)
    documents = sum(1 for f in files if f.status == "embedded")
    chunks = sum(f.chunk_count or 0 for f in files)
    synced = [f.last_synced for f in files if f.last_synced]
    last_sync = max(synced).isoformat() if synced else None
    errors = [
        {"file": f.file_name, "reason": f.error or f.status}
        for f in files
        if f.status in ("error", "no_extractable_text")
    ]
    return {
        "documents": documents,
        "chunks": chunks,
        "last_sync": last_sync,
        "errors": errors,
        "files": [
            {
                "file_id": f.file_id,
                "file_name": f.file_name,
                "status": f.status,
                "chunk_count": f.chunk_count,
                "last_synced": f.last_synced.isoformat() if f.last_synced else None,
            }
            for f in files
        ],
    }
