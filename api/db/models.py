"""SQLAlchemy models: user configuration and per-file sync state.

Stored in SQLite (a file on the app-data volume), separate from the vector store.
Secrets persisted in `Config` (Anthropic key, service-account JSON) MUST be masked
in API responses and never logged raw; the DB file is gitignored so they're never
committed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Config(Base):
    """Single-row (id=1) application configuration set via the Settings UI.

    Secret fields are persisted but must be masked whenever returned to a client.
    """

    __tablename__ = "config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    drive_folder_id: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String, default="all-MiniLM-L6-v2")
    chat_model: Mapped[str] = mapped_column(String, default="claude-sonnet-4-6")
    top_k: Mapped[int] = mapped_column(Integer, default=5)

    # ---- Secrets: masked in responses, never logged raw, never committed ----
    anthropic_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    service_account_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class FileRecord(Base):
    """Per-file sync state — the source of truth for the incremental diff.

    Drive file ids are stable across renames/edits; md5_checksum changes when the
    bytes change. The diff (Phase 2) reconciles a live Drive listing against these
    rows into added / modified / deleted / unchanged sets.
    """

    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String, primary_key=True)  # Drive id (stable)
    file_name: Mapped[str] = mapped_column(String)
    md5_checksum: Mapped[str | None] = mapped_column(String, nullable=True)
    modified_time: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # status: pending | embedded | error | no_extractable_text
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_synced: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ChatMessage(Base):
    """A single turn in a chat session (multi-turn history, keyed by session_id).

    Stores the raw Q/A text only — retrieved chunks are NOT re-stored per turn;
    we re-retrieve fresh context each turn (see prompt_builder / §7 of the plan).
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
