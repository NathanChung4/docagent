"""End-to-end ingestion test against the sample domain."""

from __future__ import annotations

from collections import Counter

from knowledge_rag.models import SourceType


def test_ingest_all_returns_expected_documents(sample_docs) -> None:
    # 6 wiki + 6 code + 4 reports + 10 checklist rows = 26
    assert len(sample_docs) == 26

    counts = Counter(d.source_type for d in sample_docs)
    assert counts[SourceType.WIKI] == 6
    assert counts[SourceType.CODE] == 6
    assert counts[SourceType.REPORT] == 4
    assert counts[SourceType.CHECKLIST] == 10


def test_ingested_docs_have_unique_ids(sample_docs) -> None:
    ids = [d.doc_id for d in sample_docs]
    assert len(set(ids)) == len(ids), "all doc_ids must be unique"


def test_ingested_docs_have_resolvable_uris(sample_docs) -> None:
    for d in sample_docs:
        assert d.uri
