"""FastAPI app: ingestion, streaming query, sessions, documents, analytics.

Layout: one router per concern, all mounted on `/api`. Async-native throughout —
the agent and Anthropic client are async, the Postgres pool is async, and
sync work (loaders, chunkers, BM25 rebuild, vector-store CRUD) runs in
`asyncio.to_thread` so a slow ingest doesn't stall live query traffic.

Bootstrap order in `lifespan`:
  1. Open the asyncpg-style pool, create the sessions/query_logs schema.
  2. Build the Embedder, VectorStore, Reranker, BM25Index. Bootstrap BM25
     from whatever's already in the vector store.
  3. Build the AsyncAgent on top of the active domain's tools.
On shutdown: close the pool, close the vector-store connection.

The `/api/query` endpoint streams Server-Sent Events. Event names match the
agent stream: `token`, `tool_call`, `tool_result`, `done`. Each event's data
is a JSON blob.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from knowledge_rag.agent import (
    AgentResult,
    AsyncAgent,
    ToolCallEvent,
    ToolResultEvent,
)
from knowledge_rag.chunking.base import chunk_documents
from knowledge_rag.domain import get_domain
from knowledge_rag.embeddings import Embedder
from knowledge_rag.loaders.ingestion import ingest_all
from knowledge_rag.models import (
    SSE_EVENT_DONE,
    SSE_EVENT_TOKEN,
    SSE_EVENT_TOOL_CALL,
    SSE_EVENT_TOOL_RESULT,
    QueryLog,
    Session,
    Turn,
)
from knowledge_rag.persistence import (
    Stats,
    append_turns,
    compute_stats,
    create_session,
    get_session,
    init_schema,
    list_query_logs,
    make_pool,
    save_query_log,
)
from knowledge_rag.reranker import CrossEncoderReranker
from knowledge_rag.retrieval import BM25Index, Retriever
from knowledge_rag.vectorstore import DEFAULT_DSN, DEFAULT_TABLE, VectorStore

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


# --- shared app state --------------------------------------------------------


@dataclass
class AppState:
    """Bundle of long-lived resources held on `app.state.deps`."""

    pool: Any  # AsyncConnectionPool — Any to avoid import-time dep
    embedder: Embedder
    vector_store: VectorStore
    bm25_index: BM25Index
    retriever: Retriever
    agent: AsyncAgent

    async def aclose(self) -> None:
        await self.pool.close()
        # Vector store owns a sync psycopg connection — close in a thread so
        # we don't block the event loop on the libpq teardown.
        await asyncio.to_thread(self.vector_store.close)


def _settings() -> dict[str, str]:
    """Pull configuration from env vars; fall back to local-dev defaults.

    KNOWLEDGE_DB_DSN is the canonical setting (used by docker-compose, .env.example,
    and the eval scripts). VECTOR_DSN / SESSION_DSN remain as per-store overrides
    for the case where a deployment splits the vector and session stores onto
    separate Postgres instances.
    """
    base_dsn = os.environ.get("KNOWLEDGE_DB_DSN", DEFAULT_DSN)
    return {
        "vector_dsn": os.environ.get("VECTOR_DSN", base_dsn),
        "session_dsn": os.environ.get("SESSION_DSN", base_dsn),
        "vector_table": os.environ.get("VECTOR_TABLE", DEFAULT_TABLE),
    }


def _bootstrap_sync_components(
    settings: dict[str, str],
) -> tuple[Embedder, VectorStore, BM25Index, Retriever]:
    """Build the sync RAG components and seed BM25 from existing vector data."""
    embedder = Embedder()
    store = VectorStore(
        dsn=settings["vector_dsn"],
        table_name=settings["vector_table"],
        embedder=embedder,
    )
    # Bootstrap BM25 by re-running the active domain's loaders + chunker.
    # Running on disk is cheap and avoids a "fetch all chunks from pgvector"
    # query that the store doesn't expose.
    domain = get_domain()
    docs = ingest_all(domain)
    chunks = chunk_documents(docs)
    bm25 = BM25Index()
    bm25.rebuild(chunks)

    reranker: CrossEncoderReranker | None
    try:
        reranker = CrossEncoderReranker()
    except Exception:
        # Reranker model download can fail offline — degrade gracefully.
        log.warning("CrossEncoderReranker unavailable; running without rerank.", exc_info=True)
        reranker = None
    retriever = Retriever(vector_store=store, bm25_index=bm25, reranker=reranker)
    return embedder, store, bm25, retriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = _settings()

    pool = await make_pool(settings["session_dsn"])
    await init_schema(pool)

    embedder, store, bm25, retriever = await asyncio.to_thread(_bootstrap_sync_components, settings)
    domain = get_domain()
    agent = AsyncAgent(tools=domain.tools())

    app.state.deps = AppState(
        pool=pool,
        embedder=embedder,
        vector_store=store,
        bm25_index=bm25,
        retriever=retriever,
        agent=agent,
    )
    try:
        yield
    finally:
        await app.state.deps.aclose()


def get_state(request: Request) -> AppState:
    return request.app.state.deps  # type: ignore[no-any-return]


# --- request/response schemas -----------------------------------------------


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    history: list[dict[str, Any]]


class QueryLogResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    session_id: str | None
    cost_usd: float
    latency_ms: float
    first_token_ms: float
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    tool_calls: list[dict[str, Any]]
    timestamp: str

    @classmethod
    def from_log(cls, log: QueryLog) -> QueryLogResponse:
        return cls(
            query_id=log.query_id,
            query=log.query,
            answer=log.answer,
            session_id=log.session_id,
            cost_usd=log.cost_usd,
            latency_ms=log.latency_ms,
            first_token_ms=log.first_token_ms,
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
            cached_tokens=log.cached_tokens,
            tool_calls=[
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "success": tc.success,
                    "error": tc.error,
                }
                for tc in log.tool_calls
            ],
            timestamp=log.timestamp.isoformat(),
        )


class StatsResponse(BaseModel):
    query_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_first_token_ms: float
    total_cost_usd: float
    cache_hit_rate: float
    tool_call_counts: dict[str, int]

    @classmethod
    def from_stats(cls, s: Stats) -> StatsResponse:
        return cls(**s.to_dict())


# --- helpers ----------------------------------------------------------------


def _session_to_response(session: Session) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        history=[
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in session.history
        ],
    )


def _sse_event(event: str, data: Any) -> dict[str, str]:
    """sse-starlette event payload. `data` is JSON-encoded."""
    return {"event": event, "data": json.dumps(data, default=str)}


# --- app + routes -----------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="Knowledge RAG API", lifespan=lifespan)

    @app.post("/api/ingest", response_model=IngestResponse)
    async def ingest(state: AppState = Depends(get_state)) -> IngestResponse:
        """Re-run loaders + chunker, upsert into pgvector, rebuild BM25."""
        domain = get_domain()

        def _do_ingest() -> tuple[int, int]:
            docs = ingest_all(domain)
            chunks = chunk_documents(docs)
            state.vector_store.add_chunks(chunks)
            state.bm25_index.rebuild(chunks)
            return len(docs), len(chunks)

        n_docs, n_chunks = await asyncio.to_thread(_do_ingest)
        return IngestResponse(documents=n_docs, chunks=n_chunks)

    @app.post("/api/sessions", response_model=SessionResponse, status_code=201)
    async def post_session(state: AppState = Depends(get_state)) -> SessionResponse:
        session = await create_session(state.pool)
        return _session_to_response(session)

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session_route(
        session_id: str, state: AppState = Depends(get_state)
    ) -> SessionResponse:
        session = await get_session(state.pool, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return _session_to_response(session)

    @app.post("/api/query")
    async def query(
        body: QueryRequest, state: AppState = Depends(get_state)
    ) -> EventSourceResponse:
        """Stream the agent's answer as SSE: token / tool_call / tool_result / done."""
        session: Session | None = None
        if body.session_id:
            session = await get_session(state.pool, body.session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")

        # Retrieval is sync — run off-thread so the event loop stays free.
        results = await asyncio.to_thread(state.retriever.retrieve, body.query, body.k)

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            final_log: QueryLog | None = None
            try:
                async for item in state.agent.stream(body.query, results, session=session):
                    if isinstance(item, str):
                        yield _sse_event(SSE_EVENT_TOKEN, {"text": item})
                    elif isinstance(item, ToolCallEvent):
                        yield _sse_event(
                            SSE_EVENT_TOOL_CALL,
                            {
                                "tool_name": item.tool_name,
                                "args": item.args,
                                "tool_use_id": item.tool_use_id,
                            },
                        )
                    elif isinstance(item, ToolResultEvent):
                        yield _sse_event(
                            SSE_EVENT_TOOL_RESULT,
                            {
                                "tool_name": item.tool_name,
                                "tool_use_id": item.tool_use_id,
                                "success": item.success,
                                "error": item.error,
                                "result": item.result,
                            },
                        )
                    elif isinstance(item, AgentResult):
                        final_log = item.query_log
                        yield _sse_event(
                            SSE_EVENT_DONE,
                            {
                                "query_id": item.query_log.query_id,
                                "cost_usd": item.query_log.cost_usd,
                                "latency_ms": item.query_log.latency_ms,
                                "first_token_ms": item.query_log.first_token_ms,
                                "input_tokens": item.query_log.input_tokens,
                                "output_tokens": item.query_log.output_tokens,
                                "cached_tokens": item.query_log.cached_tokens,
                                "iterations": item.iterations,
                                "sources": [
                                    {
                                        "chunk_id": r.chunk.chunk_id,
                                        "title": r.chunk.title,
                                        "uri": r.chunk.uri,
                                        "score": r.score,
                                    }
                                    for r in results
                                ],
                            },
                        )
            finally:
                # Persist whatever we have, even on client disconnect — partial
                # answers are still useful for analytics and session history.
                if final_log is not None:
                    await save_query_log(state.pool, final_log)
                    if session is not None:
                        await append_turns(
                            state.pool,
                            session.session_id,
                            [
                                Turn(role="user", content=body.query),
                                Turn(role="assistant", content=final_log.answer),
                            ],
                        )

        return EventSourceResponse(event_stream())

    @app.get("/api/documents")
    async def list_docs_route(state: AppState = Depends(get_state)) -> list[dict[str, Any]]:
        return await asyncio.to_thread(state.vector_store.list_documents)

    @app.delete("/api/documents/{doc_id}", status_code=204)
    async def delete_doc_route(doc_id: str, state: AppState = Depends(get_state)) -> None:
        def _delete_and_rebuild() -> None:
            state.vector_store.delete_by_doc_id(doc_id)
            domain = get_domain()
            docs = ingest_all(domain)
            chunks = chunk_documents([d for d in docs if d.doc_id != doc_id])
            state.bm25_index.rebuild(chunks)

        await asyncio.to_thread(_delete_and_rebuild)

    @app.get("/api/queries", response_model=list[QueryLogResponse])
    async def list_queries_route(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        session_id: str | None = None,
        state: AppState = Depends(get_state),
    ) -> list[QueryLogResponse]:
        logs = await list_query_logs(state.pool, session_id=session_id, limit=limit, offset=offset)
        return [QueryLogResponse.from_log(log) for log in logs]

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats_route(state: AppState = Depends(get_state)) -> StatsResponse:
        s = await compute_stats(state.pool)
        return StatsResponse.from_stats(s)

    return app


# `app` exists at module level for `uvicorn knowledge_rag.api:app`.
app = create_app()
