"""Sanity tests for the dataclass models."""

from __future__ import annotations

from knowledge_rag.models import (
    Chunk,
    Document,
    QueryLog,
    Session,
    SourceType,
    ToolCall,
    Turn,
)


def test_document_auto_id_and_timestamp() -> None:
    doc = Document(
        source_type=SourceType.WIKI,
        title="t",
        content="c",
        uri="u",
    )
    assert doc.doc_id  # auto-generated
    assert len(doc.doc_id) == 12
    assert doc.loaded_at is not None
    assert doc.metadata == {}


def test_chunk_inherits_doc_metadata_explicitly() -> None:
    chunk = Chunk(
        doc_id="abc",
        content="hello",
        source_type=SourceType.CODE,
        title="my_script",
        uri="/x.py",
    )
    assert chunk.embedding is None
    assert chunk.chunk_id != "abc"


def test_session_history_starts_empty() -> None:
    s = Session()
    assert s.history == []
    s.history.append(Turn(role="user", content="hi"))
    assert len(s.history) == 1


def test_querylog_defaults() -> None:
    q = QueryLog(query="what is foo?")
    assert q.answer == ""
    assert q.cost_usd == 0.0
    assert q.tool_calls == []


def test_toolcall_defaults() -> None:
    tc = ToolCall(tool_name="foo", args={"x": 1})
    assert tc.success is True
    assert tc.error is None


def test_source_type_serializes_to_string() -> None:
    # Inheriting from str makes JSON-encoding trivial.
    assert SourceType.WIKI == "wiki"
    assert SourceType.REPORT.value == "report"


def test_chunk_from_document_inherits_doc_fields() -> None:
    doc = Document(
        source_type=SourceType.WIKI,
        title="Page Title",
        content="full body",
        uri="mem://page",
        metadata={"author": "alice"},
    )
    chunk = Chunk.from_document(doc, "slice of body", {"section": "Intro"})
    assert chunk.doc_id == doc.doc_id
    assert chunk.source_type == doc.source_type
    assert chunk.title == doc.title
    assert chunk.uri == doc.uri
    assert chunk.content == "slice of body"
    assert chunk.metadata == {"section": "Intro"}


def test_chunk_to_filter_dict_flattens_metadata() -> None:
    chunk = Chunk(
        doc_id="d1",
        content="x",
        source_type=SourceType.CODE,
        title="t",
        uri="u",
        metadata={"feature": "alpha"},
    )
    flat = chunk.to_filter_dict()
    assert flat["doc_id"] == "d1"
    assert flat["source_type"] == "code"
    assert flat["feature"] == "alpha"
