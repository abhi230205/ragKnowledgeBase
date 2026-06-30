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
from datetime import datetime, timezone

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


def sync_state() -> dict:
    """Fetch the live/last sync-job state (running flag, last auto-sync time, ...)."""
    try:
        return api_get("/sync/status").json()
    except requests.RequestException:
        return {}


def _relative(iso: str | None) -> str:
    """Render a UTC ISO timestamp as a timezone-agnostic relative time.

    Both the stored timestamp and 'now' are UTC, so the elapsed delta is correct
    regardless of the viewer's local timezone (avoids the UTC-vs-local confusion).
    """
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return f"{secs // 86400} d ago"


def _iter_sse(path: str, timeout: int = 600):
    """Yield (event, data) tuples from a server SSE endpoint (event:/data: lines)."""
    with requests.get(f"{API_URL}{path}", stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        event = None
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    continue
                yield event, data


def run_sync_with_progress() -> None:
    """POST /sync, then render a live progress bar by consuming /sync/stream.

    202 starts a run; 409 means one is already running (we still attach to show its
    progress); 422 means Drive/creds aren't configured. Reruns at the end so the
    'Last sync' caption refreshes.
    """
    try:
        r = api_post("/sync")
    except requests.RequestException as exc:
        st.error(f"Sync failed: {exc}")
        return
    if r.status_code == 422:
        st.warning(r.json().get("error", "Configure Drive folder + service account first."))
        return
    if r.status_code == 409:
        st.info("A sync is already running — showing its progress.")
    elif r.status_code != 202:
        st.error(f"Sync failed ({r.status_code}): {r.text[:300]}")
        return

    bar = st.progress(0.0, text="Starting sync…")
    try:
        for event, data in _iter_sse("/sync/stream"):
            if event == "progress":
                total = data.get("total") or 0
                done = data.get("done") or 0
                if total > 0:
                    phase = (data.get("phase") or "working").title()
                    fname = data.get("file") or ""
                    bar.progress(min(1.0, done / total), text=f"{phase} {done}/{total} — {fname}")
                else:
                    bar.progress(0.0, text="Checking for changes…")
            elif event == "done":
                # toasts survive the rerun below (st.error/success would be wiped out)
                if data.get("running"):
                    # stream hit its safety cap while the job was still going
                    bar.progress(1.0, text="Still syncing…")
                    st.toast("Sync still running — use ↻ to check status.", icon="⏳")
                    break
                bar.progress(1.0, text="Sync complete")
                err = data.get("last_error")
                if err:
                    st.toast(f"Sync error: {err}", icon="⚠️")
                else:
                    s = (data.get("summary") or {}).get("summary") or {}
                    file_errors = s.get("errors") or []
                    st.toast(
                        f"Synced — +{s.get('added', 0)} added · {s.get('modified', 0)} modified · "
                        f"{s.get('deleted', 0)} deleted · {s.get('unchanged', 0)} unchanged.",
                        icon="✅",
                    )
                    if file_errors:
                        st.toast(
                            f"⚠️ {len(file_errors)} file(s) could not be ingested — see Dashboard.",
                            icon="⚠️",
                        )
                break
    except requests.RequestException as exc:
        st.error(f"Lost connection to sync stream: {exc}")
        return
    st.rerun()


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
    st.caption("Sync is triggered from the **Chat** page; auto-sync also runs in the background.")
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
    c3.metric("Last sync", _relative(status.get("last_sync")))

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

    # --- sync controls (Sync now + refresh adjacent), last-sync line below ---
    state = sync_state()
    sc = st.columns([1.2, 0.6, 3.2, 1.0])
    do_sync = False
    with sc[0]:
        if st.button("🔄 Sync now", type="primary"):
            do_sync = True
    with sc[1]:
        st.button("↻", help="Refresh sync status")  # any click reruns the script
    with sc[3]:
        if st.button("New chat"):
            st.session_state.session_id = "ui_" + uuid.uuid4().hex[:8]
            st.session_state.messages = []
            st.rerun()

    # Live progress bar (full width, below the buttons) while a manual sync runs.
    if do_sync:
        run_sync_with_progress()

    # "Last sync" reflects the most recent completed sync — manual OR auto.
    if state.get("running"):
        st.caption("⏳ Sync in progress…")
    else:
        mins = state.get("auto_sync_minutes", 15)
        st.caption(
            f"🕒 Last sync: {_relative(state.get('finished_at'))} · auto-sync every {mins} min"
        )
    st.caption(f"Session: `{st.session_state.session_id}`")

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
