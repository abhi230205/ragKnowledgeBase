"""SQLAlchemy engine + session factory for the SQLite config/state store.

The engine is created lazily-friendly (no filesystem I/O at import); the parent
directory and tables are created in init_db(), called on app startup. This keeps
`import main` cheap for tests that don't touch the DB.
"""

from __future__ import annotations

import os

from config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

# check_same_thread=False so the APScheduler background job (a different thread)
# can use sessions from the same engine.
engine = create_engine(
    f"sqlite:///{settings.sqlite_path}",
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db() -> None:
    """Create the SQLite parent dir and all tables (idempotent)."""
    db_dir = os.path.dirname(settings.sqlite_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new ORM session. The caller is responsible for closing it."""
    return SessionLocal()
