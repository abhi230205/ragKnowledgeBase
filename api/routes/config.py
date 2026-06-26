"""Configuration routes (GET/POST /config).

TODO (Phase 1/5): persist Drive folder id, Anthropic key, service-account JSON,
embedding model and top_k into the SQLite `config` table.

Security (graded):
- Mask every secret in responses, e.g. "sk-ant-…••••" and service_account="uploaded ✓".
- Never echo raw secrets back; never log them; never put them in URL query params.

Planned response shape (GET, masked):
    {"folder_id": "...", "embedding_model": "all-MiniLM-L6-v2", "top_k": 5,
     "anthropic_key": "sk-ant-…••••", "service_account": "uploaded ✓"}
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/config", tags=["config"])


@router.get("")
def get_config():
    """Return current config with secrets masked. TODO: implement (Phase 1)."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "GET /config not implemented yet (Phase 1)"},
    )


@router.post("")
def save_config():
    """Persist config (folder id, key, SA JSON, model, top_k). TODO (Phase 1)."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "POST /config not implemented yet (Phase 1)"},
    )
