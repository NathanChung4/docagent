"""Sync HTTP client for the Knowledge RAG FastAPI backend.

Streamlit reruns the script top-to-bottom on every user interaction, so a
sync client is the natural fit — there's no event loop to share with the
backend, and `st.write_stream` consumes a sync generator. The client uses
`httpx.Client` for short-lived requests and `httpx.stream` for the SSE
query endpoint.

The streaming endpoint yields typed events (`TokenEvent`, `ToolCallEvent`,
`ToolResultEvent`, `DoneEvent`) so the chat page can render each kind
differently without re-parsing JSON in the UI layer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from knowledge_rag.models import (
    SSE_EVENT_DONE,
    SSE_EVENT_TOKEN,
    SSE_EVENT_TOOL_CALL,
    SSE_EVENT_TOOL_RESULT,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0  # ingestion + multi-turn agent runs can be slow


def base_url() -> str:
    return os.environ.get("KNOWLEDGE_RAG_API_URL", DEFAULT_BASE_URL)


# --- typed SSE events -------------------------------------------------------


@dataclass
class TokenEvent:
    text: str


@dataclass
class ToolCallEvent:
    tool_name: str
    args: dict[str, Any]
    tool_use_id: str


@dataclass
class ToolResultEvent:
    tool_name: str
    tool_use_id: str
    success: bool
    error: str | None = None
    result: Any = None


@dataclass
class DoneEvent:
    query_id: str
    cost_usd: float
    latency_ms: float
    first_token_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    iterations: int
    sources: list[dict[str, Any]]


StreamEvent = TokenEvent | ToolCallEvent | ToolResultEvent | DoneEvent


def _parse_sse_event(event: str, data_str: str) -> StreamEvent | None:
    """Decode one SSE frame into a typed event. Unknown events return None."""
    data = json.loads(data_str)
    if event == SSE_EVENT_TOKEN:
        return TokenEvent(text=data["text"])
    if event == SSE_EVENT_TOOL_CALL:
        return ToolCallEvent(
            tool_name=data["tool_name"],
            args=data["args"],
            tool_use_id=data["tool_use_id"],
        )
    if event == SSE_EVENT_TOOL_RESULT:
        return ToolResultEvent(
            tool_name=data["tool_name"],
            tool_use_id=data["tool_use_id"],
            success=data["success"],
            error=data.get("error"),
            result=data.get("result"),
        )
    if event == SSE_EVENT_DONE:
        return DoneEvent(
            query_id=data["query_id"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
            first_token_ms=data["first_token_ms"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cached_tokens=data["cached_tokens"],
            iterations=data["iterations"],
            sources=data["sources"],
        )
    return None


# --- client -----------------------------------------------------------------


class APIClient:
    def __init__(
        self,
        base: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base = (base or base_url()).rstrip("/")
        self._timeout = timeout
        # `transport` is the seam used by tests to inject httpx.MockTransport
        # without monkey-patching the global httpx.Client.
        self._transport = transport

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(timeout=timeout or self._timeout, transport=self._transport)

    # --- sessions -----------------------------------------------------------

    def create_session(self) -> dict[str, Any]:
        with self._client() as c:
            r = c.post(f"{self._base}/api/sessions")
            r.raise_for_status()
            return r.json()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._client() as c:
            r = c.get(f"{self._base}/api/sessions/{session_id}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    # --- documents ----------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        with self._client() as c:
            r = c.get(f"{self._base}/api/documents")
            r.raise_for_status()
            return r.json()

    def delete_document(self, doc_id: str) -> None:
        with self._client() as c:
            r = c.delete(f"{self._base}/api/documents/{doc_id}")
            r.raise_for_status()

    def ingest(self) -> dict[str, int]:
        # Re-ingest can be slow on cold caches — generous timeout for this call.
        with self._client(timeout=600.0) as c:
            r = c.post(f"{self._base}/api/ingest")
            r.raise_for_status()
            return r.json()

    # --- analytics ----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._client() as c:
            r = c.get(f"{self._base}/api/stats")
            r.raise_for_status()
            return r.json()

    def list_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if session_id:
            params["session_id"] = session_id
        with self._client() as c:
            r = c.get(f"{self._base}/api/queries", params=params)
            r.raise_for_status()
            return r.json()

    # --- streaming query ----------------------------------------------------

    def stream_query(
        self,
        query: str,
        session_id: str | None = None,
        k: int = 5,
    ) -> Iterator[StreamEvent]:
        """Yield typed events as they arrive from /api/query."""
        body: dict[str, Any] = {"query": query, "k": k}
        if session_id:
            body["session_id"] = session_id

        with self._client() as c:
            with c.stream("POST", f"{self._base}/api/query", json=body) as r:
                r.raise_for_status()
                event_name: str | None = None
                data_lines: list[str] = []
                for line in r.iter_lines():
                    if line == "":
                        if event_name and data_lines:
                            parsed = _parse_sse_event(event_name, "\n".join(data_lines))
                            if parsed is not None:
                                yield parsed
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        # SSE comment line (often a heartbeat) — not a data event.
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
