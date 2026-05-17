"""Core data models used throughout the RAG pipeline.

Every object that flows between components — loaders, chunker, retriever, generator,
agent — is one of the dataclasses defined here. Keep this module dependency-free
(stdlib only) so it can be imported from anywhere without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4


class SourceType(StrEnum):
    """Inherits from StrEnum so it serializes cleanly to JSON without custom encoders."""

    WIKI = "wiki"
    CODE = "code"
    REPORT = "report"
    CHECKLIST = "checklist"


# SSE event names emitted by /api/query and consumed by the UI client.
# Defined here so backend and UI share one source of truth.
SSE_EVENT_TOKEN = "token"
SSE_EVENT_TOOL_CALL = "tool_call"
SSE_EVENT_TOOL_RESULT = "tool_result"
SSE_EVENT_DONE = "done"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    """Short unique id (first 12 hex chars of a uuid4). Readable in logs."""
    return uuid4().hex[:12]


@dataclass
class Document:
    """A whole document loaded from one upstream source file.

    A document is the unit of ingestion: one wiki page, one Python file, one CSV
    report, one row of a checklist. Documents are later split into Chunks for
    embedding and retrieval.
    """

    source_type: SourceType
    title: str
    content: str
    uri: str
    doc_id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=_utc_now)


@dataclass
class Chunk:
    """A retrievable slice of a Document.

    Chunks carry enough inherited fields (doc_id, title, uri) to be cited in
    an answer, plus per-chunk metadata (section, function name, row range)
    used for filtering during retrieval.
    """

    doc_id: str
    content: str
    source_type: SourceType
    title: str
    uri: str
    chunk_id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @classmethod
    def from_document(
        cls,
        doc: Document,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Build a Chunk inheriting doc_id/source_type/title/uri from `doc`.

        The single source of truth for "what does a chunk inherit from its
        parent document" — chunkers should never construct Chunk(...) directly.
        """
        return cls(
            doc_id=doc.doc_id,
            content=content,
            source_type=doc.source_type,
            title=doc.title,
            uri=doc.uri,
            metadata=dict(metadata) if metadata else {},
        )

    def to_filter_dict(self) -> dict[str, Any]:
        """Flat projection of this chunk's filterable fields.

        Same shape as `where` clauses passed to retrievers — keeps semantic
        and BM25 retrieval applying filters against the same field layout.
        """
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type.value,
            "title": self.title,
            "uri": self.uri,
            **self.metadata,
        }


@dataclass
class RetrievalResult:
    """One hit returned by a retriever.

    `score` is the upstream score (semantic for VectorStore, blended for
    HybridRetriever). `rerank_score` is set by the cross-encoder reranker;
    consumers that mix reranked and non-reranked results should sort on
    `rerank_score if not None else score`.
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None


@dataclass
class Turn:
    """One user/assistant exchange inside a Session."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass
class Session:
    """Multi-turn conversation state, persisted server-side.

    Sessions let follow-up questions ("what about its --thresh param?") see
    prior turns. The agent loop stays stateless; state lives here.
    """

    session_id: str = field(default_factory=_new_id)
    history: list[Turn] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass
class ToolCall:
    """A single tool invocation made by the agent during a query.

    Recorded for observability and for the Phase 9 tool-call accuracy metric.
    """

    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    success: bool = True


@dataclass
class QueryLog:
    """Per-query observability record.

    Captures everything needed to answer 'what did the system do, was it good,
    and what did it cost?' for one query. Persisted to enable analytics.
    """

    query: str
    answer: str = ""
    session_id: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    first_token_ms: float = 0.0
    query_id: str = field(default_factory=_new_id)
    timestamp: datetime = field(default_factory=_utc_now)
