"""Streamlit UI for the RAG Knowledge Base (stub — built out in Phase 5).

Talks to the FastAPI service over HTTP (base URL from the API_URL env var); the
chat page will read the /chat SSE stream into st.write_stream for token-by-token
output. Planned pages:
  - Settings : Drive folder id, Anthropic key, service-account JSON upload,
               embedding-model select, top_k (secrets shown masked).
  - Dashboard: sync trigger + live status (documents, chunks, last sync, errors).
  - Chat     : streaming chat with a citation panel mapping [n] -> file + page.

This Day-1 stub just verifies connectivity to the API's /health route.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")
st.title("📚 RAG Knowledge Base")
st.caption("Day-1 scaffold — Settings · Dashboard · Chat arrive in later phases.")

st.subheader("API connectivity")
try:
    resp = requests.get(f"{API_URL}/health", timeout=5)
    if resp.ok and resp.json().get("status") == "ok":
        st.success(f"Connected to API at {API_URL} ✓")
    else:
        st.warning(f"API responded with {resp.status_code}: {resp.text}")
except requests.RequestException as exc:
    st.error(f"Could not reach API at {API_URL}: {exc}")

with st.sidebar:
    st.header("Navigation (coming soon)")
    st.radio("Page", ["Settings", "Dashboard", "Chat"], index=0, disabled=True)
