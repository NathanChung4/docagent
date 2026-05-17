"""Persistence layer tests against the testcontainer pgvector fixture.

The pg_container/pg_dsn fixtures are session-scoped (3s startup amortized).
Each test creates a fresh pool, lets `init_schema` create tables on a unique
prefix, and tears it down. We don't isolate per-test schemas — the queries
all carry their own session_id / query_id so cross-test bleed is impossible.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from knowledge_rag.models import QueryLog, ToolCall, Turn
from knowledge_rag.persistence import (
    append_turns,
    compute_stats,
    create_session,
    get_query_log,
    get_session,
    init_schema,
    list_query_logs,
    make_pool,
    save_query_log,
)


@pytest.fixture
async def pool(pg_dsn):
    """Fresh schema, fresh pool per test. Drops the tables on teardown."""
    p = await make_pool(pg_dsn)
    await init_schema(p)
    # Cleanup any data from prior tests sharing this container.
    async with p.connection() as conn:
        await conn.execute("TRUNCATE sessions, query_logs RESTART IDENTITY CASCADE;")
    try:
        yield p
    finally:
        await p.close()


# --- sessions ----------------------------------------------------------------


async def test_create_and_get_session(pool) -> None:
    session = await create_session(pool)
    assert session.session_id
    fetched = await get_session(pool, session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id
    assert fetched.history == []


async def test_get_session_missing(pool) -> None:
    assert await get_session(pool, "no_such_session") is None


async def test_append_turns_extends_history(pool) -> None:
    session = await create_session(pool)
    turns = [
        Turn(role="user", content="hi"),
        Turn(role="assistant", content="hello"),
    ]
    await append_turns(pool, session.session_id, turns)

    refreshed = await get_session(pool, session.session_id)
    assert refreshed is not None
    assert len(refreshed.history) == 2
    assert refreshed.history[0].role == "user"
    assert refreshed.history[0].content == "hi"
    assert refreshed.history[1].role == "assistant"

    # Append-only semantics: a second call extends, not replaces.
    await append_turns(pool, session.session_id, [Turn(role="user", content="again")])
    refreshed = await get_session(pool, session.session_id)
    assert len(refreshed.history) == 3


async def test_append_turns_empty_is_noop(pool) -> None:
    session = await create_session(pool)
    await append_turns(pool, session.session_id, [])
    refreshed = await get_session(pool, session.session_id)
    assert refreshed.history == []


# --- query logs --------------------------------------------------------------


def _make_log(query: str = "q", session_id: str | None = None, **overrides) -> QueryLog:
    return QueryLog(
        query=query,
        answer=overrides.get("answer", "a"),
        session_id=session_id,
        retrieved_chunk_ids=overrides.get("retrieved_chunk_ids", ["c1", "c2"]),
        tool_calls=overrides.get("tool_calls", []),
        input_tokens=overrides.get("input_tokens", 100),
        output_tokens=overrides.get("output_tokens", 50),
        cached_tokens=overrides.get("cached_tokens", 200),
        cache_creation_tokens=overrides.get("cache_creation_tokens", 0),
        cost_usd=overrides.get("cost_usd", 0.001),
        latency_ms=overrides.get("latency_ms", 250.0),
        first_token_ms=overrides.get("first_token_ms", 80.0),
    )


async def test_save_and_get_query_log(pool) -> None:
    log = _make_log()
    await save_query_log(pool, log)
    fetched = await get_query_log(pool, log.query_id)

    assert fetched is not None
    assert fetched.query == "q"
    assert fetched.answer == "a"
    assert fetched.retrieved_chunk_ids == ["c1", "c2"]
    assert fetched.input_tokens == 100
    assert fetched.cost_usd == pytest.approx(0.001)


async def test_save_query_log_persists_tool_calls(pool) -> None:
    log = _make_log(
        tool_calls=[
            ToolCall(tool_name="echo", args={"x": 1}, result={"y": 2}, success=True),
            ToolCall(tool_name="other", args={}, error="bad", success=False),
        ]
    )
    await save_query_log(pool, log)
    fetched = await get_query_log(pool, log.query_id)

    assert len(fetched.tool_calls) == 2
    assert fetched.tool_calls[0].tool_name == "echo"
    assert fetched.tool_calls[0].result == {"y": 2}
    assert fetched.tool_calls[1].success is False
    assert fetched.tool_calls[1].error == "bad"


async def test_save_query_log_upserts_on_conflict(pool) -> None:
    log = _make_log(answer="first")
    await save_query_log(pool, log)
    log.answer = "updated"
    await save_query_log(pool, log)

    fetched = await get_query_log(pool, log.query_id)
    assert fetched.answer == "updated"


async def test_query_log_with_session_fk(pool) -> None:
    session = await create_session(pool)
    log = _make_log(session_id=session.session_id)
    await save_query_log(pool, log)

    fetched = await get_query_log(pool, log.query_id)
    assert fetched.session_id == session.session_id


async def test_list_query_logs_orders_by_recency(pool) -> None:
    earliest = _make_log(query="oldest")
    earliest.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    middle = _make_log(query="middle")
    middle.timestamp = datetime(2026, 2, 1, tzinfo=UTC)
    latest = _make_log(query="newest")
    latest.timestamp = datetime(2026, 3, 1, tzinfo=UTC)

    for log in (earliest, middle, latest):
        await save_query_log(pool, log)

    logs = await list_query_logs(pool, limit=10)
    assert [log.query for log in logs] == ["newest", "middle", "oldest"]


async def test_list_query_logs_respects_session_filter(pool) -> None:
    session = await create_session(pool)
    log_with = _make_log(query="with-session", session_id=session.session_id)
    log_without = _make_log(query="without-session")
    await save_query_log(pool, log_with)
    await save_query_log(pool, log_without)

    scoped = await list_query_logs(pool, session_id=session.session_id)
    assert len(scoped) == 1
    assert scoped[0].query == "with-session"


# --- analytics ---------------------------------------------------------------


async def test_compute_stats_empty_db(pool) -> None:
    stats = await compute_stats(pool)
    assert stats.query_count == 0
    assert stats.avg_latency_ms == 0
    assert stats.tool_call_counts == {}


async def test_compute_stats_aggregates_correctly(pool) -> None:
    for i in range(3):
        log = _make_log(
            query=f"q{i}",
            latency_ms=100.0 * (i + 1),
            cost_usd=0.001 * (i + 1),
            cached_tokens=200,
            input_tokens=100,
            tool_calls=[ToolCall(tool_name="echo", args={}, success=True)] if i > 0 else [],
        )
        await save_query_log(pool, log)

    stats = await compute_stats(pool)
    assert stats.query_count == 3
    assert stats.avg_latency_ms == pytest.approx(200.0)  # (100+200+300)/3
    assert stats.total_cost_usd == pytest.approx(0.006)
    assert stats.tool_call_counts == {"echo": 2}
    # cache_hit_rate = sum(cached) / sum(cached + input) = 600 / (600 + 300)
    assert stats.cache_hit_rate == pytest.approx(600 / 900)
