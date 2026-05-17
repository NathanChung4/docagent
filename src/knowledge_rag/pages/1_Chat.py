"""Chat page — multi-turn streaming Q&A with tool-use visibility.

Streamlit reruns the script top-to-bottom on every interaction, so the
chat history lives in `st.session_state.messages` and is replayed on each
rerun. New messages stream live: tokens fill an `st.empty()` placeholder
incrementally, tool calls render as `st.status` blocks inline, and the
final `done` event populates a sources expander and a metrics caption.

The session_id returned by POST /api/sessions is also stored in
`st.session_state` and sent with every query, so the backend (Postgres-
backed `sessions` table) is the source of truth for what the agent sees.
The local `messages` list is purely for rendering.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from knowledge_rag.ui import get_client
from knowledge_rag.ui.api_client import (
    APIClient,
    DoneEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

st.set_page_config(page_title="Chat — Knowledge RAG", layout="wide")
st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: {role, content, tool_calls?, sources?, metrics?}

if "session_id" not in st.session_state:
    st.session_state.session_id = None

client: APIClient = get_client()


def _ensure_session() -> str:
    """Lazily create a backend session on first message."""
    if st.session_state.session_id is None:
        sess = client.create_session()
        st.session_state.session_id = sess["session_id"]
    return st.session_state.session_id


# --- sidebar controls ------------------------------------------------------

with st.sidebar:
    st.markdown("### Session")
    st.code(st.session_state.session_id or "(none yet)", language="text")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.markdown("### Retrieval")
    top_k = st.slider("top-k chunks", min_value=1, max_value=20, value=5)


# --- render past messages --------------------------------------------------


def _render_tool_call(call: dict[str, Any]) -> None:
    """Render a completed tool call as a collapsed status block."""
    label_kind = "Done" if call["success"] else "Failed"
    state = "complete" if call["success"] else "error"
    with st.status(f"{label_kind}: `{call['tool_name']}`", state=state, expanded=False):
        st.markdown("**Arguments**")
        st.json(call["args"])
        if call["success"]:
            st.markdown("**Result**")
            result = call.get("result")
            if isinstance(result, (dict, list)):
                st.json(result)
            else:
                st.code(str(result))
        else:
            st.error(call.get("error") or "Tool failed without an error message.")


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            score = s.get("score")
            score_str = f" — score {score:.3f}" if isinstance(score, (int, float)) else ""
            st.markdown(f"- **{s['title']}**{score_str}  \n  `{s['uri']}`")


def _render_metrics(m: dict[str, Any]) -> None:
    cached_pct = 100.0 * m["cached_tokens"] / m["input_tokens"] if m["input_tokens"] else 0.0
    st.caption(
        f"{m['latency_ms']:.0f}ms total · "
        f"{m['first_token_ms']:.0f}ms to first token · "
        f"${m['cost_usd']:.4f} · "
        f"{m['iterations']} agent turn(s) · "
        f"{m['input_tokens']} in / {m['output_tokens']} out tokens · "
        f"{cached_pct:.0f}% cache hit"
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        for call in msg.get("tool_calls", []):
            _render_tool_call(call)
        if msg.get("content"):
            st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))
        if msg.get("metrics"):
            _render_metrics(msg["metrics"])


# --- new message input + streaming render ----------------------------------

prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    session_id = _ensure_session()

    with st.chat_message("assistant"):
        # Use a container so we can append placeholders in stream order:
        # tokens fill the current text placeholder; a tool call freezes it
        # and starts a new tool block + new text placeholder for what
        # comes next. This preserves the natural narrative ordering of
        # "agent talks → calls tool → talks more".
        container = st.container()
        text_buffer = ""
        text_placeholder = container.empty()
        pending_tools: dict[str, dict[str, Any]] = {}  # tool_use_id -> partial dict
        completed_tools: list[dict[str, Any]] = []
        text_segments: list[str] = []  # all text pieces, joined for history
        sources: list[dict[str, Any]] = []
        metrics: dict[str, Any] | None = None

        try:
            for event in client.stream_query(prompt, session_id=session_id, k=top_k):
                if isinstance(event, TokenEvent):
                    text_buffer += event.text
                    text_placeholder.markdown(text_buffer)
                elif isinstance(event, ToolCallEvent):
                    # Freeze whatever text we have, then open a tool status
                    # block, then reserve a new placeholder for tokens that
                    # may follow the tool result.
                    if text_buffer:
                        text_segments.append(text_buffer)
                        text_buffer = ""
                    tool_status_ph = container.empty()
                    with tool_status_ph.container():
                        with st.status(
                            f"Calling `{event.tool_name}`", state="running", expanded=True
                        ):
                            st.json(event.args)
                    pending_tools[event.tool_use_id] = {
                        "tool_name": event.tool_name,
                        "args": event.args,
                        "placeholder": tool_status_ph,
                    }
                    text_placeholder = container.empty()
                elif isinstance(event, ToolResultEvent):
                    pending = pending_tools.pop(event.tool_use_id, None)
                    if pending is None:
                        continue
                    tool_record: dict[str, Any] = {
                        "tool_name": pending["tool_name"],
                        "args": pending["args"],
                        "success": event.success,
                        "error": event.error,
                        "result": event.result,
                    }
                    completed_tools.append(tool_record)
                    # Replace the "Calling..." block with the final state.
                    pending["placeholder"].empty()
                    with pending["placeholder"].container():
                        _render_tool_call(tool_record)
                elif isinstance(event, DoneEvent):
                    sources = event.sources
                    metrics = {
                        "query_id": event.query_id,
                        "cost_usd": event.cost_usd,
                        "latency_ms": event.latency_ms,
                        "first_token_ms": event.first_token_ms,
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "cached_tokens": event.cached_tokens,
                        "iterations": event.iterations,
                    }
        except Exception as exc:  # noqa: BLE001 — surface backend errors in chat
            st.error(f"Backend error: {exc}")

        # Flush trailing text and render sources + metrics inside this turn.
        if text_buffer:
            text_segments.append(text_buffer)
        _render_sources(sources)
        if metrics:
            _render_metrics(metrics)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "\n\n".join(s for s in text_segments if s),
            "tool_calls": completed_tools,
            "sources": sources,
            "metrics": metrics,
        }
    )
