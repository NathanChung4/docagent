"""Streamlit UI package — chat, knowledge browser, analytics."""

from __future__ import annotations

import streamlit as st

from knowledge_rag.ui.api_client import APIClient


def get_client() -> APIClient:
    """Return the per-session APIClient, instantiating it on first access.

    Stored in `st.session_state` so the same instance survives Streamlit's
    script reruns; pages just call this instead of repeating the init guard.
    """
    if "client" not in st.session_state:
        st.session_state.client = APIClient()
    return st.session_state.client  # type: ignore[no-any-return]
