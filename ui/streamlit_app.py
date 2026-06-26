"""Streamlit UI for the RAG Knowledge Base.

Three pages (sidebar): Settings · Dashboard · Chat. Talks to the FastAPI service
over HTTP; the chat page reads the /chat SSE stream into st.write_stream for
token-by-token output and renders a citation panel. Secrets are entered here and
stored server-side (SQLite); they are shown masked and never echoed back.
"""

from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
TIMEOUT = 15

st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")


# ----------------------------------------------------------------- API helpers


def api_get(path: str) -> requests.Response:
    return requests.get(f"{API_URL}{path}", timeout=TIMEOUT)


def api_post(path: str, payload: dict | None = None) -> requests.Response:
    return requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)


def check_health() -> bool:
    try:
        r = api_get("/health")
        return r.ok and r.json().get("status") == "ok"
    except requests.RequestException:
        return False


def render_citations(cites: list[dict]) -> None:
    if not cites:
        return
    with st.expander(f"📑 Sources ({len(cites)})"):
        for c in cites:
            page = c.get("page")
            page_str = f"p.{page}" if page is not None else "p.?"
            st.markdown(f"**[{c.get('id')}]** {c.get('file_name')} — {page_str}")


# ----------------------------------------------------------------- Settings


def page_settings() -> None:
    st.header("⚙️ Settings")
    try:
        cfg = api_get("/config").json()
    except requests.RequestException as exc:
        st.error(f"Could not load config: {exc}")
        return

    with st.form("settings"):
        folder = st.text_input("Google Drive folder ID", value=cfg.get("drive_folder_id") or "")
        top_k = st.number_input(
            "Top-k retrieval", min_value=1, max_value=20, value=int(cfg.get("top_k") or 5)
        )
        chat_model = st.text_input("Chat model", value=cfg.get("chat_model") or "claude-sonnet-4-6")
        st.caption(
            f"Embedding model: **{cfg.get('embedding_model')}** "
            "(changing it requires a full re-index — not hot-swappable yet)."
        )

        st.markdown("**Secrets** — stored server-side, shown masked; leave blank to keep current.")
        st.caption(f"Anthropic API key: {cfg.get('anthropic_key') or '— not set —'}")
        api_key = st.text_input("New Anthropic API key", type="password", placeholder="sk-ant-…")
        st.caption(f"Service account: {cfg.get('service_account') or '— not uploaded —'}")
        sa_file = st.file_uploader("Upload service-account JSON", type=["json"])

        submitted = st.form_submit_button("Save settings", type="primary")

    if submitted:
        payload: dict = {
            "drive_folder_id": folder,
            "top_k": int(top_k),
            "chat_model": chat_model,
        }
        if api_key:
            payload["anthropic_api_key"] = api_key
        if sa_file is not None:
            try:
                payload["service_account_json"] = sa_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.error("Service-account file is not valid UTF-8 JSON.")
                return
        try:
            r = api_post("/config", payload)
        except requests.RequestException as exc:
            st.error(f"Save failed: {exc}")
            return
        if r.ok:
            st.success("Settings saved.")
        else:
            detail = (
                r.json().get("detail")
                if r.headers.get("content-type", "").startswith("application/json")
                else r.text
            )
            st.error(f"Save failed ({r.status_code}): {detail}")


# ----------------------------------------------------------------- Dashboard


def page_dashboard() -> None:
    st.header("📊 Dashboard")

    left, right = st.columns([1, 1])
    with left:
        if st.button("🔄 Sync now", type="primary"):
            try:
                r = api_post("/sync")
                if r.status_code == 202:
                    st.success(f"Sync started (job {r.json().get('job_id')}).")
                elif r.status_code == 409:
                    st.warning("A sync is already running.")
                else:
                    st.error(f"Sync failed ({r.status_code}): {r.text[:300]}")
            except requests.RequestException as exc:
                st.error(f"Sync failed: {exc}")
    with right:
        st.button("↻ Refresh status")  # any button click reruns the script

    try:
        status = api_get("/status").json()
    except requests.RequestException as exc:
        st.error(f"Could not load status: {exc}")
        return

    sync = status.get("sync") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", status.get("documents", 0))
    c2.metric("Chunks", status.get("chunks", 0))
    c3.metric("Last sync", status.get("last_sync") or "—")

    if sync.get("running"):
        st.info(f"Sync running… (job {sync.get('job_id')}). Click *Refresh status*.")
    elif sync.get("last_error"):
        st.error(f"Last sync error: {sync['last_error']}")

    if sync.get("last_summary"):
        with st.expander("Last sync summary"):
            st.json(sync["last_summary"])

    errors = status.get("errors") or []
    if errors:
        st.subheader("File errors")
        st.table(errors)

    files = status.get("files") or []
    if files:
        st.subheader("Documents")
        st.dataframe(files, use_container_width=True)
    else:
        st.caption(
            "No documents yet — set a Drive folder ID + upload the service-account "
            "JSON in **Settings**, then click **Sync now**."
        )


# ----------------------------------------------------------------- Chat


def stream_chat(session_id: str, message: str):
    """Return (generator, holder). The generator yields token text for
    st.write_stream; holder is filled with citations / error during iteration.
    top_k is omitted — the API uses the saved config value."""
    holder: dict = {"citations": [], "error": None}

    def gen():
        try:
            with requests.post(
                f"{API_URL}/chat",
                json={"session_id": session_id, "message": message},
                stream=True,
                timeout=120,
            ) as resp:
                event = None
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("event:"):
                        event = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        try:
                            payload = json.loads(line[len("data:") :].strip())
                        except json.JSONDecodeError:
                            continue
                        if event == "token":
                            yield payload.get("text", "")
                        elif event == "citations":
                            holder["citations"] = payload
                        elif event == "error":
                            holder["error"] = payload.get("message", "error")
                            yield f"\n\n⚠️ {holder['error']}"
        except requests.RequestException as exc:
            holder["error"] = str(exc)
            yield f"\n\n⚠️ Could not reach the chat API: {exc}"

    return gen, holder


def page_chat() -> None:
    st.header("💬 Chat")
    st.caption("Answers are grounded in your synced documents, with citations.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = "ui_" + uuid.uuid4().hex[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []

    cols = st.columns([4, 1])
    cols[0].caption(f"Session: `{st.session_state.session_id}`")
    if cols[1].button("New chat"):
        st.session_state.session_id = "ui_" + uuid.uuid4().hex[:8]
        st.session_state.messages = []
        st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            render_citations(m.get("citations") or [])

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            gen, holder = stream_chat(st.session_state.session_id, prompt)
            answer = st.write_stream(gen())
            citations = holder["citations"]
            render_citations(citations)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )


# ----------------------------------------------------------------- layout

st.sidebar.title("📚 RAG Knowledge Base")
if check_health():
    st.sidebar.success("API connected")
    healthy = True
else:
    st.sidebar.error(f"API unreachable at {API_URL}")
    healthy = False

page = st.sidebar.radio("Navigate", ["Chat", "Dashboard", "Settings"])

if not healthy:
    st.warning("Cannot reach the API. Make sure the stack is running (`docker compose up`).")
elif page == "Chat":
    page_chat()
elif page == "Dashboard":
    page_dashboard()
else:
    page_settings()
