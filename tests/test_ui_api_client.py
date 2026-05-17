"""Tests for the Streamlit UI's HTTP/SSE client.

Focuses on the SSE parser since that's the only real logic — the request
wrappers are thin enough to verify by reading. Uses an `httpx.MockTransport`
to feed a canned SSE byte stream into `stream_query` and asserts the typed
events come out in order.
"""

from __future__ import annotations

import json

import httpx

from knowledge_rag.ui.api_client import (
    APIClient,
    DoneEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    _parse_sse_event,
)

# --- SSE frame parser -------------------------------------------------------


def test_parse_sse_event_token() -> None:
    ev = _parse_sse_event("token", json.dumps({"text": "hello"}))
    assert isinstance(ev, TokenEvent)
    assert ev.text == "hello"


def test_parse_sse_event_tool_call() -> None:
    ev = _parse_sse_event(
        "tool_call",
        json.dumps({"tool_name": "lookup", "args": {"x": 1}, "tool_use_id": "tu_1"}),
    )
    assert isinstance(ev, ToolCallEvent)
    assert ev.tool_name == "lookup"
    assert ev.args == {"x": 1}
    assert ev.tool_use_id == "tu_1"


def test_parse_sse_event_tool_result_success() -> None:
    ev = _parse_sse_event(
        "tool_result",
        json.dumps(
            {
                "tool_name": "lookup",
                "tool_use_id": "tu_1",
                "success": True,
                "error": None,
                "result": {"status": "ok"},
            }
        ),
    )
    assert isinstance(ev, ToolResultEvent)
    assert ev.success is True
    assert ev.error is None
    assert ev.result == {"status": "ok"}


def test_parse_sse_event_tool_result_failure() -> None:
    ev = _parse_sse_event(
        "tool_result",
        json.dumps(
            {
                "tool_name": "lookup",
                "tool_use_id": "tu_1",
                "success": False,
                "error": "boom",
                "result": None,
            }
        ),
    )
    assert isinstance(ev, ToolResultEvent)
    assert ev.success is False
    assert ev.error == "boom"


def test_parse_sse_event_done() -> None:
    payload = {
        "query_id": "q1",
        "cost_usd": 0.0042,
        "latency_ms": 1234.5,
        "first_token_ms": 321.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 80,
        "iterations": 2,
        "sources": [{"chunk_id": "c1", "title": "Doc", "uri": "u", "score": 0.9}],
    }
    ev = _parse_sse_event("done", json.dumps(payload))
    assert isinstance(ev, DoneEvent)
    assert ev.query_id == "q1"
    assert ev.iterations == 2
    assert ev.sources[0]["title"] == "Doc"


def test_parse_sse_event_unknown_returns_none() -> None:
    assert _parse_sse_event("unknown_event", "{}") is None


# --- end-to-end stream over mock transport ----------------------------------


def _sse_bytes(*frames: tuple[str, dict]) -> bytes:
    """Render frames in SSE wire format: `event: NAME\\ndata: JSON\\n\\n`."""
    parts = []
    for event_name, data in frames:
        parts.append(f"event: {event_name}\ndata: {json.dumps(data)}\n\n")
    return "".join(parts).encode("utf-8")


def test_stream_query_yields_events_in_order() -> None:
    body_bytes = _sse_bytes(
        ("token", {"text": "Hello "}),
        ("token", {"text": "world"}),
        ("tool_call", {"tool_name": "lookup", "args": {"x": 1}, "tool_use_id": "tu_1"}),
        (
            "tool_result",
            {
                "tool_name": "lookup",
                "tool_use_id": "tu_1",
                "success": True,
                "error": None,
                "result": {"answer": 42},
            },
        ),
        ("token", {"text": " done"}),
        (
            "done",
            {
                "query_id": "q1",
                "cost_usd": 0.001,
                "latency_ms": 100.0,
                "first_token_ms": 10.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "cached_tokens": 0,
                "iterations": 1,
                "sources": [],
            },
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/query"
        sent = json.loads(request.content)
        assert sent["query"] == "hi"
        assert sent["session_id"] == "s_42"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body_bytes,
        )

    transport = httpx.MockTransport(handler)
    client = APIClient(base="http://test", transport=transport)
    events = list(client.stream_query("hi", session_id="s_42"))

    assert [type(e).__name__ for e in events] == [
        "TokenEvent",
        "TokenEvent",
        "ToolCallEvent",
        "ToolResultEvent",
        "TokenEvent",
        "DoneEvent",
    ]
    assert events[0].text == "Hello "  # type: ignore[union-attr]
    assert events[2].tool_name == "lookup"  # type: ignore[union-attr]
    assert events[3].success is True  # type: ignore[union-attr]
    assert events[5].query_id == "q1"  # type: ignore[union-attr]
