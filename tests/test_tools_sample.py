"""Sample-domain tool tests.

Exercises the three real tool implementations against the sample data shipped
in `data/sample/`. No mocks — the tools read the actual sample CSVs and xlsx
so we catch path / parsing regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domains.sample.tools.generate_config_file import GenerateConfigFile
from domains.sample.tools.lookup_item_status import LookupItemStatus
from domains.sample.tools.summarize_report import SummarizeReport
from knowledge_rag.tools.base import ToolValidationError

# --- generate_config_file ----------------------------------------------------


def test_generate_config_file_rejects_unknown_component(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="Unknown component"):
        tool.run(item_name="not_a_real_component", params={"foo": 1})


def test_generate_config_file_rejects_unknown_param(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="Unknown parameter"):
        tool.run(item_name="flow_controller", params={"made_up": 5})


def test_generate_config_file_rejects_out_of_range(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="outside allowed range"):
        # rate_limit max is 1000 per the wiki spec.
        tool.run(item_name="flow_controller", params={"rate_limit": 9999})


def test_generate_config_file_rejects_wrong_type(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="must be int"):
        tool.run(item_name="flow_controller", params={"rate_limit": 100.5})


def test_generate_config_file_rejects_bool_for_int(tmp_path: Path) -> None:
    """bool is an int subclass — guard against True passing as 1."""
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError):
        tool.run(item_name="flow_controller", params={"rate_limit": True})


def test_generate_config_file_rejects_empty_params(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="No parameters"):
        tool.run(item_name="flow_controller", params={})


def test_generate_config_file_writes_csv(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    out = tool.run(
        item_name="flow_controller",
        params={"rate_limit": 200, "burst_size": 50},
    )

    assert out["status"] == "ok"
    assert out["item_name"] == "flow_controller"
    written = Path(out["output_path"])
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "param,value" in text
    assert "rate_limit,200" in text
    assert "burst_size,50" in text


def test_generate_config_file_validates_enum(tmp_path: Path) -> None:
    tool = GenerateConfigFile(output_dir=tmp_path)
    with pytest.raises(ToolValidationError, match="must be one of"):
        tool.run(item_name="signal_buffer", params={"drop_policy": "garbage"})

    out = tool.run(item_name="signal_buffer", params={"drop_policy": "newest"})
    assert out["status"] == "ok"


# --- summarize_report --------------------------------------------------------


def test_summarize_report_picks_winner(paths) -> None:
    tool = SummarizeReport(sweep_reports_dir=paths.sweep_reports_dir)
    out = tool.run(item_name="flow_controller")

    assert out["status"] == "ok"
    assert out["report_file"] == "flow_controller_sweep.csv"
    assert out["metric"] == "throughput"
    assert out["direction"] == "max"
    # The fixture sweep tops out at rate_limit=500 -> throughput=494.2.
    assert out["winning_run"]["throughput"] == pytest.approx(494.2)
    assert out["run_count"] == 8
    assert "rate_limit" in out["param_ranges"]


def test_summarize_report_supports_metric_override(paths) -> None:
    tool = SummarizeReport(sweep_reports_dir=paths.sweep_reports_dir)
    out = tool.run(item_name="flow_controller", metric="latency_p99_ms", direction="min")
    assert out["winning_run"]["latency_p99_ms"] == pytest.approx(52.1)


def test_summarize_report_unknown_component(paths) -> None:
    tool = SummarizeReport(sweep_reports_dir=paths.sweep_reports_dir)
    with pytest.raises(ToolValidationError, match="No sweep report"):
        tool.run(item_name="ghost_component")


def test_summarize_report_rejects_unknown_metric(paths) -> None:
    tool = SummarizeReport(sweep_reports_dir=paths.sweep_reports_dir)
    with pytest.raises(ToolValidationError, match="not present"):
        tool.run(item_name="flow_controller", metric="nonexistent_metric")


# --- lookup_item_status ------------------------------------------------------


def test_lookup_item_status_finds_known_item(paths) -> None:
    tool = LookupItemStatus(checklist_path=paths.checklist_path)
    out = tool.run(item_name="voltage_monitor")
    assert out["found"] is True
    assert out["fields"]["owner"] == "jamie.t"
    assert out["fields"]["status"] == "blocked"


def test_lookup_item_status_case_insensitive(paths) -> None:
    tool = LookupItemStatus(checklist_path=paths.checklist_path)
    out = tool.run(item_name="VOLTAGE_MONITOR")
    assert out["found"] is True


def test_lookup_item_status_missing(paths) -> None:
    tool = LookupItemStatus(checklist_path=paths.checklist_path)
    out = tool.run(item_name="not_in_the_list")
    assert out["found"] is False


def test_lookup_item_status_rejects_empty(paths) -> None:
    tool = LookupItemStatus(checklist_path=paths.checklist_path)
    with pytest.raises(ToolValidationError):
        tool.run(item_name="")


# --- domain wiring -----------------------------------------------------------


def test_domain_returns_three_tools(domain) -> None:
    """Domain.tools() must wire up all three real implementations."""
    tools = domain.tools()
    names = sorted(t.name for t in tools)
    assert names == ["generate_config_file", "lookup_item_status", "summarize_report"]


def test_tool_anthropic_schema_well_formed(domain) -> None:
    """Tool schemas must be valid Anthropic tool-use input."""
    for tool in domain.tools():
        schema = tool.to_anthropic_schema()
        assert schema["name"] == tool.name
        assert isinstance(schema["description"], str) and schema["description"]
        assert schema["input_schema"]["type"] == "object"
        assert "properties" in schema["input_schema"]
        assert "required" in schema["input_schema"]
