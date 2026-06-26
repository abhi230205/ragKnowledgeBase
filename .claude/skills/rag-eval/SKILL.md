---
name: rag-eval
description: Measure retrieval quality of the RAG knowledge base. Runs question→expected-document cases through the /search API and reports hit-rate@k, MRR, and tuning suggestions (chunk size, top_k, relevance_threshold). Use after a sync to evaluate or tune retrieval (Phase 3+).
---

# rag-eval — retrieval quality harness

Quantify whether `/search` returns the genuinely relevant chunk for known questions
(rubric: "similarity search returns genuinely relevant chunks", 15 pts) and guide
tuning. Read-only except for creating the eval template.

## Eval set
`eval/queries.json` (create if missing), shaped like:
```json
[
  {"query": "what is the refund window?", "expect_file": "policy.pdf", "expect_page": 4},
  {"query": "where is the company headquartered?", "expect_file": "about.pdf"}
]
```
`expect_page` is optional. API base URL defaults to `http://localhost:8000` (override
with `$API_URL`).

## Steps
1. If `eval/queries.json` is missing, create `eval/` + a small template and ask the
   user to fill it with questions whose answers exist in the synced corpus, then stop.
2. Check the API is up (`GET /health`) and the KB is non-empty (`GET /status` →
   `chunks > 0`). If empty, tell the user to run a sync first. If `/search` returns
   501, it isn't implemented yet (Phase 3) — say so and stop.
3. For each case: `POST /search {query, top_k}` (default `top_k=5`). A case is a HIT
   if any result's `file_name == expect_file` (and, if `expect_page` is given,
   `|page - expect_page| <= 1` to allow chunk spans). Record the rank of the first hit
   and its score.
4. Print a markdown table (query, hit?, rank, top score) plus overall **hit-rate@k**
   and **MRR** (mean reciprocal rank).
5. Recommend tuning from the misses:
   - Right doc present but ranked low → raise `top_k`, revisit chunk size, or add a
     re-ranker (bonus).
   - Right doc absent entirely → chunking/embedding issue, or the doc wasn't synced.
   - Out-of-corpus queries scoring high → `relevance_threshold` is too low.
6. Never mutate the vector store; only read via `/search` and `/status`.

## Notes
- Requires Phase 3 `/search` to be live.
- Run the stack via `docker compose`; curl the API from the host.
