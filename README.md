# RAG Knowledge Base — Google Drive + Claude

A Retrieval-Augmented Generation system that ingests PDFs from a Google Drive
folder, embeds them into an open-source vector store, and answers questions over
them in a streaming chat with citations.

> **Status:** Day 1 scaffold. The ingestion pipeline, retrieval, chat, and UI are
> built out across Phases 2–6 — see [the build plan](docs/RAG_Build_Plan_Full.md).

---

## 1. Architecture overview

Two containers, modular code inside the API:

- **`api`** (FastAPI + Uvicorn) — REST + SSE endpoints; also runs, in-process: the
  APScheduler ingestion job, the sentence-transformers embedder, **embedded
  ChromaDB** (`PersistentClient` on a volume), and SQLite (config + sync state).
- **`ui`** (Streamlit) — settings, sync dashboard, and streaming chat; calls the
  API over HTTP (chat tokens via SSE → `st.write_stream`).

```
Google Drive ──HTTPS──> Ingestion ──in-proc──> Embedder ──in-proc──> ChromaDB
                                                                        │
Streamlit UI ──HTTP/REST + SSE──> FastAPI API ──in-proc──> Retrieval ──┘
                                       │
                                       └──HTTPS+SSE──> Anthropic Messages API
SQLite (config + sync state) <──in-proc file I/O── FastAPI API
```

> A protocol-labelled diagram (mermaid) is in [docs/RAG_Build_Plan_Full.md](docs/RAG_Build_Plan_Full.md) §1 and will be inlined here.

## 2. Technology choices and justification

| Area | Choice | One-line why |
|---|---|---|
| Vector DB | **ChromaDB**, embedded | Lowest-friction open-source store with metadata filtering; no extra container. |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` (384-d) | Free, local, open-source; Claude has no embeddings endpoint. |
| Chunking | **sentence-aware sliding window**, ~800 tok / ~15% overlap, page-tracked | Overlap preserves cross-boundary context; pages enable citations. |
| Frontend | **Streamlit** | Full Python UI fast; the points are in the backend. |
| PDF | **PyMuPDF** (+ pdfplumber for tables) | Fastest, good multi-column handling. |
| Chat | **claude-sonnet-4-6**, streamed (SSE) | Required; SSE is the right fit for one-way token streaming. |

_(Full one-paragraph justifications per the rubric land here in Phase 6.)_

## 3. Setup instructions

1. **Google service account:** create one in Google Cloud, enable the Drive API,
   download the JSON key, and **share the target Drive folder with the service
   account's email**. Either drop the JSON at `./secrets/service_account.json`
   (gitignored) or upload it in the Settings UI.
2. **Environment:** `cp .env.example .env` and fill in values (or set them via the
   UI). Real secrets never go in git — only `.env.example` (placeholders) is tracked.
3. **Run:** `docker compose up --build`.
   - API: http://localhost:8000 (`/health`, `/docs`)
   - UI: http://localhost:8501

## 4. How to use the system

1. Open the UI, go to **Settings**: enter the Drive folder id, Anthropic key, and
   service-account JSON; pick the embedding model and top-k.
2. **Dashboard:** trigger **Sync** and watch status (documents, chunks, last sync).
3. **Chat:** ask questions; answers stream token-by-token with source + page
   citations. _(Built in Phases 2–5.)_

## 5. Known limitations / what I'd do differently

- Embedded Chroma & co-located ingestion favour a 5-day timeline over strict
  service isolation — Chroma server mode / Qdrant and Celery are the scale-up paths.
- Scanned/image-only PDFs are flagged (OCR via Tesseract is a possible add-on).
- _(Expanded in Phase 6.)_

## 6. Reflection on Claude Code

_(Filled in throughout: which interactions saved the most time, which suggestions
were wrong and why, and lessons on effective prompting.)_

---

## Tests

```bash
docker compose run --rm api pytest      # or, in a local venv from ./api: pytest
```
