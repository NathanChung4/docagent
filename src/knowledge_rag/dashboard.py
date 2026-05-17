"""Streamlit entrypoint for the Knowledge RAG UI.

Run with:
    streamlit run src/knowledge_rag/dashboard.py

The chat, knowledge base, and analytics surfaces live in `pages/` as
sibling files. Streamlit auto-discovers anything in `pages/` and renders
sidebar navigation; numeric prefixes (1_, 2_, 3_) control the order.

Backend URL is taken from KNOWLEDGE_RAG_API_URL (default http://localhost:8000),
so this same UI works against a local uvicorn or a deployed FastAPI service.
"""

from __future__ import annotations

import streamlit as st

from knowledge_rag.ui import get_client
from knowledge_rag.ui.api_client import base_url

st.set_page_config(
    page_title="Knowledge RAG",
    page_icon=None,
    layout="wide",
)

st.title("Knowledge RAG")
st.write(
    "Ask questions over technical documentation. Answers stream token-by-token "
    "with source citations and tool-use visibility."
)

st.sidebar.markdown("### Backend")
st.sidebar.code(base_url(), language="text")


# Cache the reachability ping so flipping between pages doesn't re-hit /api/stats
# every navigation. 30s TTL is short enough to detect a backend going down.
@st.cache_data(ttl=30)
def _reachability_check() -> dict[str, int] | None:
    try:
        return get_client().stats()
    except Exception:  # noqa: BLE001
        return None


stats = _reachability_check()
if stats is not None:
    st.sidebar.success("API reachable")
    st.sidebar.metric("Queries logged", stats["query_count"])
else:
    st.sidebar.error("API unreachable")
    st.sidebar.caption("Check that uvicorn is running and KNOWLEDGE_RAG_API_URL is set.")

st.markdown(
    """
    ### Pages
    - **Chat** — multi-turn streaming Q&A with tool use
    - **Knowledge Base** — list documents, re-ingest sources
    - **Analytics** — query stats, latency, cost, tool-call frequency
    """
)
