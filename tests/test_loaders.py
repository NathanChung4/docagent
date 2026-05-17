"""Tests for each individual loader."""

from __future__ import annotations

import pytest

from knowledge_rag.loaders.checklist import ChecklistLoader
from knowledge_rag.loaders.confluence import ConfluenceLoader
from knowledge_rag.loaders.github import GitHubLoader
from knowledge_rag.loaders.sweep_report import SweepReportLoader
from knowledge_rag.models import SourceType


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


def test_confluence_loader_loads_six_pages(confluence_docs) -> None:
    assert len(confluence_docs) == 6
    for d in confluence_docs:
        assert d.source_type == SourceType.WIKI
        assert d.title
        assert d.content
        assert "<" not in d.content
        assert isinstance(d.metadata.get("sections"), list)
        assert d.metadata["sections"], "section headers should be captured"


def test_github_loader_loads_six_scripts_with_ast_metadata(github_docs) -> None:
    assert len(github_docs) == 6
    for d in github_docs:
        assert d.source_type == SourceType.CODE
        assert d.metadata["module_docstring"]
        assert isinstance(d.metadata["functions"], list)
        assert d.metadata["functions"], "scripts define top-level functions"
        assert "parse_error" not in d.metadata, "sample scripts must parse"


def test_github_loader_records_parse_error_for_unparseable_file(tmp_path) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("def oops(:\n    pass\n", encoding="utf-8")
    docs = GitHubLoader().load(tmp_path)
    assert len(docs) == 1
    assert docs[0].metadata.get("parse_error", "").startswith("SyntaxError")


def test_sweep_report_loader_loads_four_csvs_with_rows(sweep_docs) -> None:
    assert len(sweep_docs) == 4
    for d in sweep_docs:
        assert d.source_type == SourceType.REPORT
        assert d.metadata["row_count"] > 0
        assert d.metadata["columns"]
        assert "component" in d.metadata


def test_checklist_loader_yields_one_doc_per_row(checklist_docs) -> None:
    assert len(checklist_docs) == 10
    for d in checklist_docs:
        assert d.source_type == SourceType.CHECKLIST
        assert d.metadata["item_id"]
        assert d.metadata["fields"]
        assert "owner" in d.metadata["fields"]
        assert "status" in d.metadata["fields"]


def test_loaders_return_empty_for_missing_paths(tmp_path) -> None:
    missing = tmp_path / "nope"
    assert ConfluenceLoader().load(missing) == []
    assert GitHubLoader().load(missing) == []
    assert SweepReportLoader().load(missing) == []
    assert ChecklistLoader().load(missing / "x.xlsx") == []
