"""Tests for the pgvector-backed VectorStore.

Each test gets a unique table name in the shared session-scoped Postgres
container. The store contract (add → query round-trip, delete by doc_id,
metadata round-trip, persistence) is store-agnostic; only the fixture knows
this is pgvector under the hood.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from knowledge_rag.embeddings import Embedder
from knowledge_rag.models import Chunk, Document, SourceType
from knowledge_rag.vectorstore import RetrievalResult, VectorStore

# `store` and `shared_embedder` fixtures live in `tests/conftest.py`.


def _chunk(content: str, doc_id: str = "doc-1", **meta) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        content=content,
        source_type=meta.pop("source_type", SourceType.WIKI),
        title=meta.pop("title", "Sample Page"),
        uri=meta.pop("uri", "sample://page"),
        metadata=meta,
    )


def test_add_and_count(store: VectorStore) -> None:
    n = store.add_chunks([_chunk("hello"), _chunk("world")])
    assert n == 2
    assert store.count() == 2


def test_add_empty_is_noop(store: VectorStore) -> None:
    assert store.add_chunks([]) == 0
    assert store.count() == 0


def test_query_returns_most_relevant_chunk_first(store: VectorStore) -> None:
    store.add_chunks(
        [
            _chunk("How to configure the memory controller frequency"),
            _chunk("Cookies should be baked at 350 degrees for 12 minutes"),
            _chunk("The widget supports three power modes"),
        ]
    )
    hits = store.query("memory controller config", k=3)
    assert len(hits) == 3
    assert all(isinstance(h, RetrievalResult) for h in hits)
    assert "memory controller" in hits[0].chunk.content.lower()
    # Scores monotonically non-increasing.
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_query_on_empty_collection(store: VectorStore) -> None:
    assert store.query("anything", k=5) == []


def test_query_empty_text_returns_empty(store: VectorStore) -> None:
    store.add_chunks([_chunk("something")])
    assert store.query("", k=5) == []


def test_metadata_round_trip(store: VectorStore) -> None:
    """User metadata, including non-primitive values, must survive add → query."""
    chunk = _chunk(
        "section content",
        section="Overview",
        tags=["alpha", "beta"],
        weight=0.5,
        nested={"a": 1, "b": [2, 3]},
    )
    store.add_chunks([chunk])
    hit = store.query("section content", k=1)[0]
    assert hit.chunk.metadata["section"] == "Overview"
    assert hit.chunk.metadata["tags"] == ["alpha", "beta"]
    assert hit.chunk.metadata["weight"] == 0.5
    assert hit.chunk.metadata["nested"] == {"a": 1, "b": [2, 3]}


def test_metadata_filter(store: VectorStore) -> None:
    """`where` filter narrows results to chunks whose metadata matches."""
    store.add_chunks(
        [
            _chunk("python function definition", source_type=SourceType.CODE),
            _chunk("user guide for the widget", source_type=SourceType.WIKI),
        ]
    )
    hits = store.query("definition", k=5, where={"source_type": "code"})
    assert len(hits) == 1
    assert hits[0].chunk.source_type == SourceType.CODE


def test_unsupported_where_raises(store: VectorStore) -> None:
    """Operator filters and nested filters are explicitly out of scope."""
    store.add_chunks([_chunk("anything")])
    with pytest.raises(NotImplementedError):
        store.query("anything", k=1, where={"$or": [{"a": 1}]})
    with pytest.raises(NotImplementedError):
        store.query("anything", k=1, where={"section": {"$in": ["a", "b"]}})


def test_delete_by_doc_id(store: VectorStore) -> None:
    store.add_chunks(
        [
            _chunk("a", doc_id="keep"),
            _chunk("b", doc_id="drop"),
            _chunk("c", doc_id="drop"),
        ]
    )
    assert store.count() == 3
    store.delete_by_doc_id("drop")
    assert store.count() == 1


def test_reindex_document_replaces_old_chunks(store: VectorStore) -> None:
    """Re-indexing flow: old chunks for a doc are removed, new ones added."""
    doc = Document(
        source_type=SourceType.WIKI,
        title="t",
        content="ignored",
        uri="u",
        doc_id="doc-X",
    )
    store.add_chunks([_chunk("original v1 content", doc_id="doc-X")])
    assert store.count() == 1

    store.reindex_document(
        doc,
        [
            _chunk("revised v2 chunk one", doc_id="doc-X"),
            _chunk("revised v2 chunk two", doc_id="doc-X"),
        ],
    )
    assert store.count() == 2
    hits = store.query("revised v2", k=2)
    assert all("v2" in h.chunk.content for h in hits)


def test_pre_embedded_chunks_are_not_re_embedded(
    store: VectorStore, shared_embedder: Embedder
) -> None:
    """If a chunk arrives with .embedding set, the store should reuse it."""
    chunk = _chunk("preembedded content")
    chunk.embedding = list(shared_embedder.embed_texts([chunk.content])[0])
    pre_vector = list(chunk.embedding)
    store.add_chunks([chunk])
    # Mutating the embedding after add must not affect what was stored.
    chunk.embedding = [0.0] * len(pre_vector)
    hits = store.query("preembedded content", k=1)
    assert hits[0].chunk.chunk_id == chunk.chunk_id


def test_persistence_across_instances(pg_dsn: str, shared_embedder: Embedder) -> None:
    """Two stores pointed at the same DSN+table see each other's data."""
    table = f"test_{uuid4().hex[:12]}"
    s1 = VectorStore(dsn=pg_dsn, table_name=table, embedder=shared_embedder)
    s2 = VectorStore(dsn=pg_dsn, table_name=table, embedder=shared_embedder)
    try:
        s1.add_chunks([_chunk("durable content")])
        assert s1.count() == 1
        assert s2.count() == 1
        assert s2.query("durable", k=1)[0].chunk.content == "durable content"
    finally:
        s1.reset()
        s1.close()
        s2.close()
