"""Application configuration via pydantic-settings.

Values are read from environment variables (injected by docker-compose's
`env_file: .env`). Secrets entered through the Settings UI are stored in SQLite
(see db.models.Config) and take precedence over these at runtime. Nothing here is
ever a committed secret — .env is gitignored; only .env.example (placeholders) is
tracked.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Anthropic (chat) ----
    anthropic_api_key: str | None = None
    chat_model: str = "claude-sonnet-4-6"

    # ---- Google Drive (file-based SA optional; UI upload is primary) ----
    google_service_account_path: str = "/secrets/service_account.json"
    drive_folder_id: str | None = None

    # ---- Embeddings (local, open-source) ----
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Retrieval ----
    top_k: int = 5
    relevance_threshold: float = 0.25  # cosine-similarity floor for "no context"

    # ---- Chat / context-window management ----
    max_history_turns: int = 6

    # ---- Storage (inside container; on named volumes) ----
    chroma_path: str = "/data/chroma"
    sqlite_path: str = "/data/app/rag.db"

    # ---- UI -> API ----
    api_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()


settings = get_settings()
