"""Postgres persistence for sessions and query logs.

Shares the Postgres instance that backs pgvector. Two tables:
  - sessions: one row per chat session, history as JSONB array of turns
  - query_logs: one row per query, full token / cost / tool-call telemetry

JSONB for history and tool_calls because their shape is recursive and we
rarely query into them — we read whole sessions, not individual turns.
The columns we DO query on (session_id, timestamp) are real columns with
indexes.

Async throughout via psycopg3's AsyncConnectionPool so FastAPI handlers
don't have to bounce through a threadpool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from knowledge_rag.models import QueryLog, Session, ToolCall, Turn

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    history      JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS query_logs (
    query_id              TEXT PRIMARY KEY,
    session_id            TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    query                 TEXT NOT NULL,
    answer                TEXT NOT NULL DEFAULT '',
    retrieved_chunk_ids   JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_calls            JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_tokens          INT  NOT NULL DEFAULT 0,
    output_tokens         INT  NOT NULL DEFAULT 0,
    cached_tokens         INT  NOT NULL DEFAULT 0,
    cache_creation_tokens INT  NOT NULL DEFAULT 0,
    cost_usd              DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms            DOUBLE PRECISION NOT NULL DEFAULT 0,
    first_token_ms        DOUBLE PRECISION NOT NULL DEFAULT 0,
    "timestamp"           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS query_logs_session_id_idx ON query_logs(session_id);
CREATE INDEX IF NOT EXISTS query_logs_timestamp_idx ON query_logs("timestamp" DESC);
"""


# --- pool plumbing -----------------------------------------------------------


async def make_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    """Open a pool and wait for the first connection to be ready."""
    pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size, open=False)
    await pool.open()
    await pool.wait()
    return pool


async def init_schema(pool: AsyncConnectionPool) -> None:
    """Idempotent: safe to call on every startup."""
    async with pool.connection() as conn:
        await conn.execute(DDL)


# --- session repository ------------------------------------------------------


def _turn_to_json(turn: Turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        "timestamp": turn.timestamp.isoformat(),
    }


def _session_from_row(row: dict[str, Any]) -> Session:
    history_json: list[dict[str, Any]] = row["history"] or []
    history = [
        Turn(
            role=t["role"],
            content=t["content"],
            timestamp=datetime.fromisoformat(t["timestamp"]),
        )
        for t in history_json
    ]
    return Session(
        session_id=row["session_id"],
        history=history,
        created_at=row["created_at"],
    )


async def create_session(pool: AsyncConnectionPool) -> Session:
    """Insert a fresh session and return it."""
    session = Session()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id, created_at, history) VALUES (%s, %s, %s)",
            (session.session_id, session.created_at, json.dumps([])),
        )
    return session


async def get_session(pool: AsyncConnectionPool, session_id: str) -> Session | None:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT session_id, created_at, history FROM sessions WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return _session_from_row(row)


async def append_turns(
    pool: AsyncConnectionPool,
    session_id: str,
    new_turns: list[Turn],
) -> None:
    """Append turns to the session's history JSONB. No-op if list is empty."""
    if not new_turns:
        return
    payload = json.dumps([_turn_to_json(t) for t in new_turns])
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET history = history || %s::jsonb WHERE session_id = %s",
            (payload, session_id),
        )


# --- query log repository ----------------------------------------------------


def _tool_call_to_json(tc: ToolCall) -> dict[str, Any]:
    return {
        "tool_name": tc.tool_name,
        "args": tc.args,
        "result": tc.result,
        "error": tc.error,
        "success": tc.success,
    }


def _query_log_from_row(row: dict[str, Any]) -> QueryLog:
    tc_json = row["tool_calls"] or []
    tool_calls = [
        ToolCall(
            tool_name=tc["tool_name"],
            args=tc["args"] or {},
            result=tc.get("result"),
            error=tc.get("error"),
            success=tc.get("success", True),
        )
        for tc in tc_json
    ]
    return QueryLog(
        query=row["query"],
        answer=row["answer"],
        session_id=row["session_id"],
        retrieved_chunk_ids=row["retrieved_chunk_ids"] or [],
        tool_calls=tool_calls,
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cached_tokens=row["cached_tokens"],
        cache_creation_tokens=row["cache_creation_tokens"],
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        first_token_ms=row["first_token_ms"],
        query_id=row["query_id"],
        timestamp=row["timestamp"],
    )


async def save_query_log(pool: AsyncConnectionPool, log: QueryLog) -> None:
    """Persist a QueryLog. Upsert on query_id so retries are idempotent."""
    tool_calls_json = json.dumps([_tool_call_to_json(tc) for tc in log.tool_calls])
    chunk_ids_json = json.dumps(log.retrieved_chunk_ids)
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO query_logs (
                query_id, session_id, query, answer,
                retrieved_chunk_ids, tool_calls,
                input_tokens, output_tokens, cached_tokens, cache_creation_tokens,
                cost_usd, latency_ms, first_token_ms, "timestamp"
            ) VALUES (
                %s, %s, %s, %s,
                %s::jsonb, %s::jsonb,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (query_id) DO UPDATE SET
                answer = EXCLUDED.answer,
                retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids,
                tool_calls = EXCLUDED.tool_calls,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                cached_tokens = EXCLUDED.cached_tokens,
                cache_creation_tokens = EXCLUDED.cache_creation_tokens,
                cost_usd = EXCLUDED.cost_usd,
                latency_ms = EXCLUDED.latency_ms,
                first_token_ms = EXCLUDED.first_token_ms
            """,
            (
                log.query_id,
                log.session_id,
                log.query,
                log.answer,
                chunk_ids_json,
                tool_calls_json,
                log.input_tokens,
                log.output_tokens,
                log.cached_tokens,
                log.cache_creation_tokens,
                log.cost_usd,
                log.latency_ms,
                log.first_token_ms,
                log.timestamp,
            ),
        )


async def get_query_log(pool: AsyncConnectionPool, query_id: str) -> QueryLog | None:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT * FROM query_logs WHERE query_id = %s",
                (query_id,),
            )
            row = await cur.fetchone()
    return _query_log_from_row(row) if row else None


async def list_query_logs(
    pool: AsyncConnectionPool,
    *,
    session_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[QueryLog]:
    """Most-recent-first. Optionally scoped to one session."""
    where = "WHERE session_id = %s " if session_id else ""
    params: tuple[Any, ...] = (session_id, limit, offset) if session_id else (limit, offset)
    sql = f'SELECT * FROM query_logs {where}ORDER BY "timestamp" DESC LIMIT %s OFFSET %s'
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
    return [_query_log_from_row(r) for r in rows]


# --- analytics ---------------------------------------------------------------


@dataclass
class Stats:
    """Aggregate stats for the GET /api/stats endpoint."""

    query_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_first_token_ms: float
    total_cost_usd: float
    cache_hit_rate: float
    tool_call_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def compute_stats(pool: AsyncConnectionPool) -> Stats:
    """One round-trip aggregating across all logged queries."""
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT
                    COUNT(*)                                      AS query_count,
                    COALESCE(AVG(latency_ms), 0)                  AS avg_latency_ms,
                    COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms), 0) AS p50_latency_ms,
                    COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_latency_ms,
                    COALESCE(AVG(first_token_ms), 0)              AS avg_first_token_ms,
                    COALESCE(SUM(cost_usd), 0)                    AS total_cost_usd,
                    COALESCE(SUM(cached_tokens)::float
                             / NULLIF(SUM(cached_tokens + input_tokens), 0), 0) AS cache_hit_rate
                FROM query_logs
                """
            )
            agg = await cur.fetchone()

            await cur.execute(
                """
                SELECT tc->>'tool_name' AS name, COUNT(*) AS n
                FROM query_logs, jsonb_array_elements(tool_calls) AS tc
                GROUP BY tc->>'tool_name'
                ORDER BY n DESC
                """
            )
            tool_rows = await cur.fetchall()

    return Stats(
        query_count=agg["query_count"],
        avg_latency_ms=float(agg["avg_latency_ms"]),
        p50_latency_ms=float(agg["p50_latency_ms"]),
        p95_latency_ms=float(agg["p95_latency_ms"]),
        avg_first_token_ms=float(agg["avg_first_token_ms"]),
        total_cost_usd=float(agg["total_cost_usd"]),
        cache_hit_rate=float(agg["cache_hit_rate"]),
        tool_call_counts={r["name"]: r["n"] for r in tool_rows if r["name"]},
    )
