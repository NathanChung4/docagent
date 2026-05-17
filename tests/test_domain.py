"""Tests for the Domain interface and the get_domain() factory."""

from __future__ import annotations

import pytest

from knowledge_rag.domain import DomainNotFoundError, get_domain
from knowledge_rag.tools.base import Tool


def test_get_domain_default_is_sample() -> None:
    d = get_domain()
    assert d.name == "sample"


def test_get_domain_explicit_name() -> None:
    d = get_domain("sample")
    assert d.name == "sample"


def test_get_domain_unknown_raises() -> None:
    with pytest.raises(DomainNotFoundError):
        get_domain("does_not_exist_pack")


def test_sample_paths_resolve() -> None:
    d = get_domain("sample")
    paths = d.paths()
    assert paths.confluence_dir.exists()
    assert paths.github_dir.exists()
    assert paths.sweep_reports_dir.exists()
    assert paths.checklist_path.exists()


def test_sample_tools_register() -> None:
    d = get_domain("sample")
    tools = d.tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"generate_config_file", "summarize_report", "lookup_item_status"}
    for t in tools:
        assert isinstance(t, Tool)
        assert t.description
        assert t.input_schema["type"] == "object"


def test_sample_tools_round_trip_to_anthropic_schema() -> None:
    d = get_domain("sample")
    for t in d.tools():
        schema = t.to_anthropic_schema()
        assert set(schema.keys()) == {"name", "description", "input_schema"}


def test_sample_eval_dataset_loads() -> None:
    d = get_domain("sample")
    dataset = d.eval_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) >= 1
    # Each entry must have at minimum an id, question, and kind.
    for entry in dataset:
        assert "id" in entry
        assert "question" in entry
        assert entry["kind"] in {"qa", "tool"}
