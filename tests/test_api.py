"""FastAPI endpoint tests against an httpx ASGI client.

Bypasses `lifespan` and injects test-built dependencies onto `app.state.deps`
so the suite stays hermetic — no Anthropic API key, no embedder weights, no
external network. The Postgres pool is the real session-scoped testcontainer
fixture from conftest.

Tested:
  - session create / fetch / 404
  - SSE stream from /api/query yields token + tool_call + done events
  - query log + session history persisted after a streaming call completes
  - analytics aggregations
  - documents list/delete dispatched to the (stubbed) vector store
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from knowledge_rag.agent import AsyncAgent
from knowledge_rag.api import AppState, create_app
from knowledge_rag.persistence import (
    init_schema,
    list_query_logs,
    make_pool,
)
from knowledge_rag.tools.base import Tool
from tests._anthropic_fakes import (
    AsyncFakeClient,
    AsyncFakeStream,
    text_block,
    tool_use_block,
    usage,
)


class _EchoTool(Tool):
    name = "echo"
    description = "echo"
    input_schema = {"type": "object", "properties": {"message": {"type": "string"}}}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return {"echoed": kwargs.get("message", "")}


@pytest.fixture
async def pool(pg_dsn):
    p = await make_pool(pg_dsn)
    await init_schema(p)
    async with p.connection() as conn:
        await conn.execute("TRUNCATE sessions, query_logs RESTART IDENTITY CASCADE;")
    try:
        yield p
    finally:
        await p.close()


def _stub_retriever(results: list = ()) -> MagicMock:
    """Stub Retriever.retrieve()-> fixed result list."""
    r = MagicMock()
    r.retrieve.return_value = list(results)
    return r


def _stub_vector_store(documents: list[dict[str, Any]] | None = None) -> MagicMock:
    s = MagicMock()
    s.list_documents.return_value = documents or []
    s.add_chunks.return_value = 0
    s.delete_by_doc_id.return_value = None
    s.close.return_value = None
    return s


@pytest.fixture
async def app_with_state(pool):
    """FastAPI app pre-populated with test deps; no lifespan run."""
    app = create_app()

    fake_client = AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=("Hello", " ", "world"),
                usage=usage(input_tokens=10, output_tokens=3),
                content=[text_block("Hello world")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = AsyncAgent(tools=[_EchoTool()], client=fake_client)

    app.state.deps = AppState(
        pool=pool,
        embedder=MagicMock(),
        vector_store=_stub_vector_store(),
        bm25_index=MagicMock(),
        retriever=_stub_retriever(),
        agent=agent,
    )
    app.state.fake_client = fake_client  # available for test assertions
    return app


@pytest.fixture
async def client(app_with_state):
    transport = httpx.ASGITransport(app=app_with_state)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# --- sessions ---------------------------------------------------------------


async def test_post_session_creates_row(client) -> None:
    resp = await client.post("/api/sessions")
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"]
    assert body["history"] == []


async def test_get_session_roundtrip(client) -> None:
    created = (await client.post("/api/sessions")).json()
    fetched = (await client.get(f"/api/sessions/{created['session_id']}")).json()
    assert fetched["session_id"] == created["session_id"]


async def test_get_session_404(client) -> None:
    resp = await client.get("/api/sessions/nonexistent")
    assert resp.status_code == 404


# --- query streaming --------------------------------------------------------


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """Parse a sse-starlette stream body into a list of {event, data} dicts."""
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current_data.append(line.split(":", 1)[1].strip())
        elif line == "" and current_event is not None:
            events.append(
                {
                    "event": current_event,
                    "data": json.loads("\n".join(current_data)) if current_data else None,
                }
            )
            current_event = None
            current_data = []
    return events


async def test_query_streams_tokens_and_done(client) -> None:
    resp = await client.post(
        "/api/query",
        json={"query": "what?", "k": 3},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    token_events = [e for e in events if e["event"] == "token"]
    done_events = [e for e in events if e["event"] == "done"]

    assert [e["data"]["text"] for e in token_events] == ["Hello", " ", "world"]
    assert len(done_events) == 1
    done = done_events[0]["data"]
    assert done["query_id"]
    assert done["input_tokens"] == 10
    assert done["output_tokens"] == 3
    assert "sources" in done


async def test_query_persists_log_to_db(client, pool) -> None:
    resp = await client.post("/api/query", json={"query": "persist me"})
    assert resp.status_code == 200
    # Drain stream so the after-stream save runs.
    _ = resp.text

    logs = await list_query_logs(pool)
    assert len(logs) == 1
    assert logs[0].query == "persist me"
    assert logs[0].answer == "Hello world"


async def test_query_appends_to_session(client, pool) -> None:
    sess = (await client.post("/api/sessions")).json()
    resp = await client.post(
        "/api/query",
        json={"query": "tied to session", "session_id": sess["session_id"]},
    )
    assert resp.status_code == 200
    _ = resp.text  # drain

    fetched = (await client.get(f"/api/sessions/{sess['session_id']}")).json()
    assert len(fetched["history"]) == 2
    assert fetched["history"][0]["role"] == "user"
    assert fetched["history"][0]["content"] == "tied to session"
    assert fetched["history"][1]["role"] == "assistant"
    assert fetched["history"][1]["content"] == "Hello world"


async def test_query_emits_tool_call_event_when_agent_calls_tool(pool) -> None:
    """End-to-end: tool_call + tool_result + token + done events all reach SSE."""
    iter1 = AsyncFakeStream(
        tokens=(),
        usage=usage(),
        content=[tool_use_block("echo", {"message": "ping"}, id_="tu_1")],
        stop_reason="tool_use",
    )
    iter2 = AsyncFakeStream(
        tokens=("Got: ", "ping"),
        usage=usage(),
        content=[text_block("Got: ping")],
        stop_reason="end_turn",
    )
    fake_client = AsyncFakeClient([iter1, iter2])
    app = create_app()
    app.state.deps = AppState(
        pool=pool,
        embedder=MagicMock(),
        vector_store=_stub_vector_store(),
        bm25_index=MagicMock(),
        retriever=_stub_retriever(),
        agent=AsyncAgent(tools=[_EchoTool()], client=fake_client),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        resp = await c.post("/api/query", json={"query": "echo it"})
        events = _parse_sse(resp.text)

    event_names = [e["event"] for e in events]
    assert "tool_call" in event_names
    assert "tool_result" in event_names
    assert "done" in event_names
    tool_call = next(e for e in events if e["event"] == "tool_call")
    assert tool_call["data"]["tool_name"] == "echo"
    assert tool_call["data"]["args"] == {"message": "ping"}


async def test_query_with_unknown_session_returns_404(client) -> None:
    resp = await client.post(
        "/api/query",
        json={"query": "nope", "session_id": "ghost"},
    )
    assert resp.status_code == 404


# --- documents --------------------------------------------------------------


async def test_list_documents_returns_stubbed_rows(client, app_with_state) -> None:
    app_with_state.state.deps.vector_store.list_documents.return_value = [
        {
            "doc_id": "d1",
            "title": "alpha",
            "source_type": "wiki",
            "uri": "/a",
            "chunk_count": 4,
        }
    ]
    resp = await client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json()[0]["doc_id"] == "d1"
    assert resp.json()[0]["chunk_count"] == 4


async def test_delete_document_dispatches_to_store(client, app_with_state) -> None:
    resp = await client.delete("/api/documents/d1")
    assert resp.status_code == 204
    app_with_state.state.deps.vector_store.delete_by_doc_id.assert_called_once_with("d1")


# --- analytics --------------------------------------------------------------


async def test_stats_empty(client) -> None:
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_count"] == 0
    assert body["tool_call_counts"] == {}


async def test_stats_after_one_query(client) -> None:
    await client.post("/api/query", json={"query": "first"})
    resp = await client.get("/api/stats")
    body = resp.json()
    assert body["query_count"] == 1
    assert body["total_cost_usd"] >= 0


async def test_queries_endpoint_lists_recent(pool) -> None:
    fake_client = AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=("a",),
                usage=usage(input_tokens=1, output_tokens=1),
                content=[text_block("a")],
                stop_reason="end_turn",
            ),
            AsyncFakeStream(
                tokens=("b",),
                usage=usage(input_tokens=1, output_tokens=1),
                content=[text_block("b")],
                stop_reason="end_turn",
            ),
        ]
    )
    app = create_app()
    app.state.deps = AppState(
        pool=pool,
        embedder=MagicMock(),
        vector_store=_stub_vector_store(),
        bm25_index=MagicMock(),
        retriever=_stub_retriever(),
        agent=AsyncAgent(tools=[_EchoTool()], client=fake_client),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r1 = await c.post("/api/query", json={"query": "first"})
        _ = r1.text  # drain so the after-stream save runs
        r2 = await c.post("/api/query", json={"query": "second"})
        _ = r2.text
        resp = await c.get("/api/queries?limit=10")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# --- ingest ------------------------------------------------------------------


async def test_ingest_returns_counts(client, app_with_state, monkeypatch) -> None:
    # Stub the loader/chunker pipeline so we don't actually touch disk.
    from knowledge_rag import api as api_mod

    monkeypatch.setattr(api_mod, "ingest_all", lambda d: ["doc1", "doc2"])
    monkeypatch.setattr(api_mod, "chunk_documents", lambda docs: ["c1", "c2", "c3"])

    resp = await client.post("/api/ingest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] == 2
    assert body["chunks"] == 3
