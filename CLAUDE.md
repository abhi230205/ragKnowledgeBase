# CLAUDE.md — RAG Knowledge Base (project context for AI sessions)

Authoritative context for this repo. Read this first. Two source-of-truth docs
live in `docs/`: the **assignment brief** (requirements/rubric — not in repo, see
the original PDF) and **`docs/RAG_Build_Plan_Full.md`** (the architecture/plan we
follow). If a request conflicts with a locked decision below, stop and ask.

## What this is
A production-quality RAG system for a 5-day graded assignment (Claude Code track).
Ingests PDFs from a Google Drive folder → embeds into an open-source vector store →
streaming chat that answers **only** from retrieved context, with citations.

## Locked stack (do NOT change without explicit approval)
- **Language/runtime:** Python 3.11+ (containers pinned to `python:3.11-slim`).
- **Backend:** FastAPI + Uvicorn (async). Streaming via **SSE**, not WebSocket.
- **Vector DB:** **ChromaDB, embedded** via `chromadb.PersistentClient(path=…)` on a
  Docker volume. NOT a separate container. (Open-source only — no managed cloud DB.)
- **Embeddings:** **local sentence-transformers**, default `all-MiniLM-L6-v2`
  (384-dim), in-process, model selectable. **Claude is NOT an embedding model —
  never embed via the Anthropic API.** (Voyage AI is the optional paid alternative.)
- **Document parsing:** PyMuPDF (`fitz`) primary for PDF; pdfplumber for table-heavy
  pages. **DOCX** via `python-docx` (`ingestion/docx_parser.py`) — paragraphs + table
  rows, returned as a single page (p.1). The pipeline dispatches by mime/extension;
  Drive lists PDF **and** DOCX (`drive_client.list_documents`, `DOC_MIMES`).
- **Chunking:** sentence-aware sliding window, **token-aware** and sized to the
  embedding model's max — default ~256 tokens for `all-MiniLM-L6-v2`, clamped at
  runtime via `embedder.effective_chunk_tokens()`; ~15% overlap; page-tracked
  (start_page/end_page). Overlap is mandatory. **Why not ~800:** MiniLM truncates
  input at 256 tokens, so larger chunks would be silently cut before embedding
  (dropping each chunk's tail). Sizing chunks to the model avoids that. Tunable via
  `settings.chunk_tokens` / `chunk_overlap`; a larger-context model raises the
  effective size automatically. (Approved change from the plan's ~800, 2026-06-26.)
- **Chat model:** `claude-sonnet-4-6`, **streamed** (Anthropic Messages API).
- **Frontend:** **Streamlit** (no React), calling the API over HTTP; chat reads the
  `/chat` SSE stream into `st.write_stream`.
- **Persistent store:** SQLite via SQLAlchemy (config + sync state) — kept separate
  from the vector store.
- **Ingestion scheduling:** APScheduler, in-process (no Celery/broker).
- **Deploy:** `docker compose up` — two services (`api`, `ui`), named volumes, one
  bridge network. Only manual step is creating `.env`.

## Hard rules (graded — violating these loses points)
- **No hardcoded secrets.** Anthropic key + service-account JSON are entered via the
  UI and stored in SQLite (or the gitignored `./secrets/` dir). `.env`, `*.db`, and
  `secrets/*` are gitignored. Committed creds = automatic 20-point deduction.
- **Mask secrets** in every API response and log (e.g. `sk-ant-…••••`). Never put
  secrets in URL query params.
- **No hallucination.** If retrieval finds nothing relevant (below
  `settings.relevance_threshold`), answer "not in the knowledge base" with empty
  citations — don't send junk context to Claude.
- **Citations required.** Every claim cites document + page; backend returns a
  structured citations list and the prompt uses `[n]` markers.
- **Incremental sync.** Track md5_checksum + modifiedTime; skip unchanged files
  (no re-embedding). On edit: delete all of a file's chunks, then re-add (avoid
  orphan chunks). On delete: remove its chunks.
- **Persistence.** Chroma + SQLite live on named volumes; restarting Docker must not
  wipe the knowledge base.
- **Modular code.** No monolithic files. Keep the package boundaries below.
- **Drive auth failure** must surface as a clear UI error, never a 500.

## Layout (packages, not a flat script)
```
api/   (FastAPI; also runs ingestion + embedder + embedded Chroma + SQLite)
  main.py            app + lifespan (init_db) + router registration
  config.py          pydantic-settings (env-driven)
  routes/            health(real) · config · sync · search · chat · status
  ingestion/         drive_client(REAL) · pdf_parser · chunker · sync_diff · scheduler
  embeddings/        embedder (sentence-transformers) · reranker (cross-encoder, bonus)
  vectorstore/       chroma_store (embedded PersistentClient, cosine)
  llm/               prompt_builder · claude_stream (Messages API streaming)
  db/                models (Config, FileRecord) · session · crud
  tests/             pytest (≥10 tests by Phase 6)
ui/    streamlit_app.py (Settings · Dashboard · Chat)
```

## Import convention
The api build context is `./api` copied to `/app`, so imports are **top-level**
(`from routes import health`, `from db.models import Config`, `from config import
settings`) — NOT prefixed with `api.`. Tests run from `api/` (`pytest.ini` sets
`pythonpath = .`).

## Conventions
- SSE event protocol for `/chat`: `token` (repeated) → `citations` → `done`; an
  `error` event on failure. Chroma cosine: relevance score = `1 - distance`.
- SSE event protocol for `/sync/stream` (live ingestion progress bar): `progress`
  (repeated, `{running,done,total,file,phase}`) → `done`
  (`{running,summary,finished_at,last_error}`; `running:true` means the stream hit its
  safety cap while the job was still going — the UI must not report that as complete).
  The pipeline reports per-file via a `progress_cb`; the scheduler stores it in
  `_state["progress"]` (reset to `None` when idle); the UI drives an `st.progress` bar.
- Deterministic Chroma ids: `"{file_id}_{chunk_index}"` (idempotent re-sync).
- Chroma metadata per chunk: `file_id, file_name, start_page, end_page,
  chunk_index, preview`.
- Run CPU-bound work (embedding, PyMuPDF) off the event loop via
  `run_in_threadpool` / `asyncio.to_thread`.
- Keep `/health` cheap and dependency-free.

## Common commands
- Build + run: `docker compose up --build`  (first build is slow — torch + chromadb).
- API: http://localhost:8000  (`/health`, `/docs`).  UI: http://localhost:8501.
- Tests: `docker compose run --rm api pytest`  (or `pytest` in a venv from `./api`).
- Logs: `docker compose logs -f api`.

## Status & roadmap (phases)
- **Phase 0–1 (Day 1) — DONE:** scaffold, Docker Compose, `/health`, pydantic config,
  SQLAlchemy models (Config/FileRecord), REAL Drive v3 client (recursive PDF list +
  byte download, typed auth/API errors), stubs for the rest, smoke test.
- **Phase 2 (Day 2) — DONE:** pdf_parser (PyMuPDF block-sorted + pdfplumber tables
  + scanned-page flagging), token-aware chunker, sentence-transformers embedder,
  embedded Chroma (cosine; upsert/query/delete/rename/count/reset), pure sync diff
  (added/modified/deleted/renamed/unchanged), APScheduler `/sync` (202, non-overlap)
  + `/sync/status`, real `/status`. Unit tests: chunking/diff/embedding/retrieval.
  Ingestion pipeline in `ingestion/pipeline.py`; per-file errors are non-fatal.
- **Phase 3 (Day 2–3):** `/search` (top-k + filter + threshold).
- **Phase 4 (Day 3):** `/chat` — grounding prompt, no-context guard, multi-turn,
  Claude streaming → SSE.
- **Phase 5 (Day 4):** Streamlit Settings + Dashboard + streaming chat with citations.
- **Phase 6 (Day 5):** edge-case sweep, ≥10 tests, README (6 sections + diagram), demo.

## Tooling (Claude Code: skills · agents · hooks · MCP)
- **Hooks** (`.claude/settings.json`): `secret_scan.py` (PreToolUse on `git commit` —
  blocks staged keys/creds, the −20 pitfall) and `ruff_format.py` (PostToolUse on
  Write|Edit — formats `.py` on save). Host needs `ruff` (`pip install ruff`; invoked
  as `python -m ruff`). Lint/format config in `ruff.toml`.
- **Custom skills** (`.claude/skills/`): `/rag-eval` (retrieval hit-rate@k / MRR over
  `/search`), `/chunk-inspect` (how a PDF chunks — drives `api/scripts/chunk_inspect.py`).
- **Custom agents** (`.claude/agents/`): `prompt-engineer` (Phase 4 grounding prompt),
  `rag-evaluator` (retrieval tuning), `edge-case-hunter` (Phase 6 adversarial sweep).
- **Built-in skills to use per phase:** `/code-review` (each phase), `/verify` + `/run`
  (after wiring routes/UI), `/security-review` (before final commit — secrets/masking).
- **MCP servers:** `playwright` (drive/screenshot the UI + demo), `sqlite` (inspect the
  config/files tables — reads a host snapshot at `.debug/rag.db`; refresh with
  `docker compose cp rag-api:/data/app/rag.db ./.debug/rag.db`), `claude.ai Google Drive`
  (seed/inspect a test corpus). MCP tools load at session start.

## Commit discipline
Atomic, descriptive commits at each logical milestone (the rubric grades commit
history). `.gitignore` was created before the first commit so secrets are never
tracked.
