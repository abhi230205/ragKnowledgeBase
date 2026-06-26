"""FastAPI application entry point for the RAG Knowledge Base API.

Wires the routers together and initialises the SQLite store on startup. Heavy
subsystems (embedder, vector store, ingestion scheduler) are loaded lazily in
later phases so the /health route stays cheap and always available.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from db.session import init_db
from ingestion.scheduler import shutdown_scheduler, start_scheduler
from routes import chat, config as config_routes, health, search, status, sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Creates DB tables (idempotent) and starts the ingestion
    scheduler. The embedding model + Chroma load lazily on first use, so /health
    stays cheap."""
    init_db()
    start_scheduler()
    logger.info(
        "RAG API started — SQLite=%s, Chroma=%s, embed=%s, chat=%s",
        settings.sqlite_path,
        settings.chroma_path,
        settings.embedding_model,
        settings.chat_model,
    )
    yield
    shutdown_scheduler()
    logger.info("RAG API shutting down.")


app = FastAPI(
    title="RAG Knowledge Base API",
    version="0.1.0",
    description="Google Drive-backed Retrieval-Augmented Generation knowledge base.",
    lifespan=lifespan,
)

# Route registration. /health is real; the rest are stubs filled in by phase.
app.include_router(health.router)
app.include_router(config_routes.router)
app.include_router(sync.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(status.router)
