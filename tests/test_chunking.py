"""Tests for the chunking pipeline.

Verifies each chunker produces semantically meaningful, well-tagged chunks for
its source type, that the dispatcher routes by source_type, and that overflow
behavior (oversized prose, large reports) is correct.
"""

from __future__ import annotations

import pytest

from knowledge_rag.chunking import (
    CodeChunker,
    ProseChunker,
    StructuredChunker,
    chunk_document,
    chunk_documents,
)
from knowledge_rag.loaders.checklist import ChecklistLoader
from knowledge_rag.loaders.confluence import ConfluenceLoader
from knowledge_rag.loaders.github import GitHubLoader
from knowledge_rag.loaders.sweep_report import SweepReportLoader
from knowledge_rag.models import Document, SourceType


@pytest.fixture(scope="module")
def confluence_docs(paths):
    return ConfluenceLoader().load(paths.confluence_dir)


@pytest.fixture(scope="module")
def github_docs(paths):
    return GitHubLoader().load(paths.github_dir)


@pytest.fixture(scope="module")
def sweep_docs(paths):
    return SweepReportLoader().load(paths.sweep_reports_dir)


@pytest.fixture(scope="module")
def checklist_docs(paths):
    return ChecklistLoader().load(paths.checklist_path)


# --- Prose chunker ----------------------------------------------------------


def test_prose_chunker_splits_wiki_by_section(confluence_docs) -> None:
    doc = next(d for d in confluence_docs if "clock" in d.title.lower())

    chunks = ProseChunker().chunk(doc)

    sections_seen = {c.metadata["section"] for c in chunks}
    expected = {"Overview", "Parameters", "Validation Procedure", "Known Issues"}
    assert expected.issubset(sections_seen)

    for c in chunks:
        assert c.source_type == SourceType.WIKI
        assert c.doc_id == doc.doc_id
        assert c.title == doc.title
        assert c.uri == doc.uri
        assert c.metadata["strategy"] == "prose"
        assert isinstance(c.metadata["chunk_index"], int)


def test_prose_chunker_keeps_section_content_with_its_header(confluence_docs) -> None:
    doc = next(d for d in confluence_docs if "clock" in d.title.lower())

    chunks = ProseChunker().chunk(doc)
    params_chunk = next(c for c in chunks if c.metadata["section"] == "Parameters")
    assert "divisor" in params_chunk.content.lower()


def test_prose_chunker_window_splits_oversized_section_with_overlap() -> None:
    long_body = "alpha beta gamma delta " * 200  # ~4600 chars
    doc = Document(
        source_type=SourceType.WIKI,
        title="Big Page",
        content=f"Big Page\nOnly Section\n{long_body}",
        uri="mem://big",
        metadata={"sections": ["Big Page", "Only Section"]},
    )
    chunker = ProseChunker(max_chars=1000, overlap_chars=100)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 1000
    # Overlap: end of chunk 0 should match start of chunk 1.
    tail = chunks[0].content[-100:]
    head = chunks[1].content[:100]
    assert tail == head


def test_prose_chunker_tolerates_doc_with_no_sections() -> None:
    doc = Document(
        source_type=SourceType.WIKI,
        title="No Headers",
        content="A flat blob of text with no section headers in it at all.",
        uri="mem://flat",
        metadata={"sections": []},
    )
    chunks = ProseChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("A flat blob")


# --- Code chunker -----------------------------------------------------------


def test_code_chunker_splits_python_by_function_and_class(github_docs) -> None:
    doc = next(d for d in github_docs if d.title == "clock_divider")

    chunks = CodeChunker().chunk(doc)
    symbols = {c.metadata["symbol"] for c in chunks}

    assert "is_power_of_two" in symbols
    assert "measure" in symbols
    assert "<module>" in symbols

    for c in chunks:
        assert c.source_type == SourceType.CODE
        assert c.metadata["strategy"] == "code"
        assert c.metadata["kind"] in {"function", "class", "module"}


def test_code_chunker_function_chunk_contains_full_function_body(github_docs) -> None:
    doc = next(d for d in github_docs if d.title == "clock_divider")

    chunks = CodeChunker().chunk(doc)
    measure_chunk = next(c for c in chunks if c.metadata["symbol"] == "measure")

    assert "def measure" in measure_chunk.content
    assert "return {" in measure_chunk.content
    assert measure_chunk.content.rstrip().endswith("}")


def test_code_chunker_falls_back_to_single_chunk_on_syntax_error() -> None:
    doc = Document(
        source_type=SourceType.CODE,
        title="broken",
        content="def oops(:\n    pass\n",
        uri="mem://broken.py",
        metadata={"filename": "broken.py", "module_docstring": "", "functions": [], "classes": []},
    )
    chunks = CodeChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["symbol"] == "<unparseable>"


# --- Structured chunker -----------------------------------------------------


def test_structured_chunker_emits_one_chunk_per_checklist_row(checklist_docs) -> None:
    chunker = StructuredChunker()
    for d in checklist_docs:
        chunks = chunker.chunk(d)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.source_type == SourceType.CHECKLIST
        assert c.metadata["strategy"] == "checklist_row"
        assert any(k.startswith("field_") for k in c.metadata)


def test_structured_chunker_batches_sweep_rows(sweep_docs) -> None:
    doc = next(d for d in sweep_docs if "clock" in d.title.lower())
    # Sample CSV has 9 rows; force small batches to verify batching logic.
    chunker = StructuredChunker(rows_per_chunk=4, row_overlap=1)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.source_type == SourceType.REPORT
        assert c.metadata["strategy"] == "report_rows"
        assert c.metadata["row_count"] <= 4
        assert "columns" in c.metadata
        assert "row " in c.content


def test_structured_chunker_overlaps_rows_between_batches() -> None:
    rows = [{"i": str(n), "v": f"v{n}"} for n in range(20)]
    columns = ["i", "v"]
    doc = Document(
        source_type=SourceType.REPORT,
        title="bench",
        content="ignored — chunker reads metadata['rows']",
        uri="mem://bench.csv",
        metadata={"rows": rows, "columns": columns, "component": "bench"},
    )
    chunker = StructuredChunker(rows_per_chunk=5, row_overlap=2)
    chunks = chunker.chunk(doc)
    # First chunk holds rows 0-4; with overlap=2, second chunk starts at row 3.
    assert chunks[0].metadata["row_start"] == 0
    assert chunks[0].metadata["row_end"] == 4
    assert chunks[1].metadata["row_start"] == 3


# --- Dispatcher -------------------------------------------------------------


def test_dispatcher_routes_by_source_type(sample_docs) -> None:
    chunks = chunk_documents(sample_docs)

    assert len(chunks) >= len(sample_docs)

    by_strategy: dict[str, set[str]] = {}
    for c in chunks:
        by_strategy.setdefault(c.metadata["strategy"], set()).add(c.source_type.value)

    assert by_strategy.get("prose") == {SourceType.WIKI.value}
    assert by_strategy.get("code") == {SourceType.CODE.value}
    assert by_strategy.get("report_rows") == {SourceType.REPORT.value}
    assert by_strategy.get("checklist_row") == {SourceType.CHECKLIST.value}


def test_dispatcher_preserves_doc_identity_on_chunks(sample_docs) -> None:
    chunks = chunk_documents(sample_docs)
    doc_ids = {d.doc_id for d in sample_docs}
    for c in chunks:
        assert c.doc_id in doc_ids
        assert c.title  # citations need this
        assert c.uri  # citations need this


def test_chunk_document_handles_unknown_source_type_gracefully() -> None:
    for st in SourceType:
        doc = Document(
            source_type=st,
            title=f"t-{st.value}",
            content="nonempty",
            uri=f"mem://{st.value}",
            metadata={"sections": [], "rows": [], "columns": [], "fields": {}},
        )
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
