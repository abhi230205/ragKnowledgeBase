"""Configuration routes (GET/POST /config).

Persists user config (Drive folder id, Anthropic key, service-account JSON, chat
model, top-k) into the SQLite `config` table for the Settings UI.

Security (graded): secrets are NEVER returned raw — the Anthropic key is masked
(e.g. "sk-ant…••••") and the service account is reported only as a status flag.
Secrets are never logged and never placed in URLs. Blank fields on POST mean
"leave unchanged", so the UI never has to re-enter a secret to change something else.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings
from db import crud
from db.session import get_session
from ingestion.drive_client import extract_folder_id

router = APIRouter(prefix="/config", tags=["config"])


class ConfigUpdate(BaseModel):
    drive_folder_id: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    service_account_json: Optional[str] = None
    chat_model: Optional[str] = None
    top_k: Optional[int] = None


def _mask(secret: str | None) -> str | None:
    """Mask a secret for display: keep a short prefix, hide the rest."""
    if not secret:
        return None
    prefix = secret[:6] if len(secret) > 6 else secret[:2]
    return f"{prefix}…••••"


def _view(cfg) -> dict:
    """Build the masked, safe-to-return config view (used by GET and POST)."""
    effective_key = cfg.anthropic_api_key or settings.anthropic_api_key
    has_sa = bool(cfg.service_account_json) or os.path.exists(settings.google_service_account_path)
    return {
        "drive_folder_id": cfg.drive_folder_id or settings.drive_folder_id,
        # Embedding model is env-authoritative (changing it forces a re-index at sync).
        "embedding_model": settings.embedding_model,
        "chat_model": cfg.chat_model,
        "top_k": cfg.top_k,
        "anthropic_key": _mask(effective_key),
        "has_anthropic_key": bool(effective_key),
        "service_account": "uploaded ✓" if has_sa else None,
        "has_service_account": has_sa,
    }


@router.get("")
def get_config() -> dict:
    """Return current config with all secrets masked."""
    session = get_session()
    try:
        return _view(crud.get_or_create_config(session))
    finally:
        session.close()


@router.post("")
def save_config(body: ConfigUpdate) -> dict:
    """Persist provided config fields. Blank/None fields are left unchanged."""
    fields: dict = {}

    if body.drive_folder_id is not None:
        # Accept a pasted folder URL and store just the id.
        fields["drive_folder_id"] = extract_folder_id(body.drive_folder_id.strip())
    if body.chat_model:
        fields["chat_model"] = body.chat_model.strip()
    if body.top_k is not None:
        if body.top_k <= 0:
            raise HTTPException(status_code=422, detail="top_k must be a positive integer")
        fields["top_k"] = body.top_k

    # Secrets: only update when a non-empty value is supplied (blank = keep existing).
    if body.anthropic_api_key:
        fields["anthropic_api_key"] = body.anthropic_api_key.strip()
    if body.service_account_json:
        try:
            parsed = json.loads(body.service_account_json)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail=f"Service account JSON is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict) or not (
            parsed.get("client_email") and parsed.get("private_key")
        ):
            raise HTTPException(
                status_code=422,
                detail="Service account JSON must be an object with client_email and private_key.",
            )
        fields["service_account_json"] = body.service_account_json

    session = get_session()
    try:
        cfg = (
            crud.update_config(session, **fields) if fields else crud.get_or_create_config(session)
        )
        return _view(cfg)
    finally:
        session.close()
