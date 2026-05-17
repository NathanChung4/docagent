"""Tests for the tool-call accuracy evaluator.

Builds synthetic AgentResult objects directly — no need to mock the Anthropic
SDK since this module operates on the post-run output (QueryLog.tool_calls),
not on the streaming loop itself.

All vocabulary stays inside the public sample-domain surface so the
no-leak guardrail doesn't flag this file.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from knowledge_rag.agent import AgentResult
from knowledge_rag.models import QueryLog, ToolCall
from knowledge_rag.tool_eval import (
    _dict_args_match,
    _values_match,
    evaluate_tool_call,
    evaluate_tool_calls,
)


def _make_agent_result(tool_calls: Sequence[ToolCall], answer: str = "") -> AgentResult:
    """Synthesize an AgentResult with the given ToolCall list."""
    return AgentResult(
        answer=answer,
        query_log=QueryLog(query="q", answer=answer, tool_calls=list(tool_calls)),
        iterations=1,
    )


# --- _values_match -----------------------------------------------------------


def test_values_match_int_string_bool_exact():
    assert _values_match(5, 5)
    assert _values_match("clock_divider", "clock_divider")
    assert _values_match(True, True)
    assert not _values_match(5, 6)
    assert not _values_match("clock_divider", "Clock_Divider")


def test_values_match_float_isclose():
    assert _values_match(0.005, 0.005)
    assert _values_match(0.005, 0.005001)
    assert not _values_match(0.005, 0.5)


def test_values_match_int_float_cross_type():
    assert _values_match(2, 2.0)


def test_values_match_lists():
    assert _values_match([1, 2, 3], [1, 2, 3])
    assert not _values_match([1, 2, 3], [1, 2])
    assert not _values_match([1, 2, 3], [3, 2, 1])


def test_values_match_recursive_dict():
    assert _values_match({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not _values_match({"a": 1}, {"a": 2})


# --- _dict_args_match --------------------------------------------------------


def test_dict_args_match_perfect_score():
    assert _dict_args_match({"x": 1, "y": 2}, {"x": 1, "y": 2}) == 1.0


def test_dict_args_match_missing_required_key():
    # Missing 1/2 keys → 0.5
    assert _dict_args_match({"x": 1, "y": 2}, {"x": 1}) == 0.5


def test_dict_args_match_wrong_value():
    # Both keys present but wrong value → 0.5
    assert _dict_args_match({"x": 1, "y": 2}, {"x": 1, "y": 99}) == 0.5


def test_dict_args_match_extra_keys_penalize():
    # All required match, but 2 extras: 1.0 - 0.2 = 0.8
    assert _dict_args_match({"x": 1}, {"x": 1, "extra1": "a", "extra2": "b"}) == pytest.approx(0.8)


def test_dict_args_match_extras_capped_at_1():
    # 100 extras shouldn't drive the score negative.
    actual = {"x": 1, **{f"e{i}": i for i in range(100)}}
    assert _dict_args_match({"x": 1}, actual) == 0.0


def test_dict_args_match_empty_expected_no_extras():
    assert _dict_args_match({}, {}) == 1.0


def test_dict_args_match_nested_dict_perfect():
    expected = {"item_name": "clock_divider", "params": {"divisor": 4, "jitter_budget_ps": 300}}
    actual = {"item_name": "clock_divider", "params": {"divisor": 4, "jitter_budget_ps": 300}}
    assert _dict_args_match(expected, actual) == 1.0


def test_dict_args_match_nested_dict_partial():
    """Nested dict that doesn't match recursively counts as a single mismatched key."""
    expected = {"item_name": "clock_divider", "params": {"divisor": 4}}
    actual = {"item_name": "clock_divider", "params": {"divisor": 999}}
    # item_name matches (1/2), params nested match is False (1/2 total).
    assert _dict_args_match(expected, actual) == 0.5


# --- evaluate_tool_call ------------------------------------------------------


def test_evaluate_tool_call_full_match():
    entry = {
        "id": "t1",
        "question": "Generate a config file for clock_divider",
        "kind": "tool",
        "expected_tool": "generate_config_file",
        "expected_tool_args": {"item_name": "clock_divider", "params": {"divisor": 4}},
    }
    agent_result = _make_agent_result(
        [
            ToolCall(
                tool_name="generate_config_file",
                args={"item_name": "clock_divider", "params": {"divisor": 4}},
                result={"status": "ok"},
                success=True,
            )
        ],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.tool_choice_correct is True
    assert verdict.args_match_score == 1.0
    assert verdict.full_match is True
    assert verdict.actual_tools_called == ["generate_config_file"]
    assert verdict.actual_validation_error is False


def test_evaluate_tool_call_wrong_tool():
    entry = {
        "id": "t2",
        "kind": "tool",
        "expected_tool": "generate_config_file",
        "expected_tool_args": {"item_name": "clock_divider"},
    }
    agent_result = _make_agent_result(
        [
            ToolCall(
                tool_name="lookup_item_status", args={"item_name": "clock_divider"}, result={"x": 1}
            )
        ],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.tool_choice_correct is False
    assert verdict.args_match_score == 0.0
    assert verdict.full_match is False


def test_evaluate_tool_call_no_tool_when_none_expected():
    entry = {"id": "t3", "kind": "tool", "expected_tool": None}
    agent_result = _make_agent_result([])
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.tool_choice_correct is True
    assert verdict.args_match_score == 1.0
    assert verdict.full_match is True


def test_evaluate_tool_call_tool_when_none_expected():
    entry = {"id": "t4", "kind": "tool", "expected_tool": None}
    agent_result = _make_agent_result(
        [ToolCall(tool_name="lookup_item_status", args={"item_name": "x"}, result={})],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.tool_choice_correct is False
    assert verdict.full_match is False


def test_evaluate_tool_call_validation_error_expected_and_present():
    entry = {
        "id": "t5",
        "kind": "tool",
        "expected_tool": "generate_config_file",
        "expected_tool_args": {"item_name": "flow_controller", "params": {"rate_limit": 50000}},
        "expected_validation_error": True,
    }
    agent_result = _make_agent_result(
        [
            ToolCall(
                tool_name="generate_config_file",
                args={"item_name": "flow_controller", "params": {"rate_limit": 50000}},
                error="rate_limit out of range",
                success=False,
            )
        ],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.tool_choice_correct is True
    assert verdict.actual_validation_error is True
    assert verdict.validation_error_correct is True


def test_evaluate_tool_call_validation_error_expected_but_missing():
    entry = {
        "id": "t6",
        "kind": "tool",
        "expected_tool": "generate_config_file",
        "expected_tool_args": {"item_name": "clock_divider"},
        "expected_validation_error": True,
    }
    agent_result = _make_agent_result(
        [
            ToolCall(
                tool_name="generate_config_file",
                args={"item_name": "clock_divider"},
                result={"status": "ok"},
                success=True,
            )
        ],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.actual_validation_error is False
    assert verdict.validation_error_correct is False


def test_evaluate_tool_call_picks_first_call_to_expected_tool_for_args():
    """If the agent calls the expected tool twice, score against the first call."""
    entry = {
        "id": "t7",
        "kind": "tool",
        "expected_tool": "lookup_item_status",
        "expected_tool_args": {"item_name": "voltage_monitor"},
    }
    agent_result = _make_agent_result(
        [
            ToolCall(
                tool_name="lookup_item_status", args={"item_name": "voltage_monitor"}, result={}
            ),
            ToolCall(tool_name="lookup_item_status", args={"item_name": "wrong_one"}, result={}),
        ],
    )
    verdict = evaluate_tool_call(entry, agent_result)
    assert verdict.args_match_score == 1.0
    assert verdict.full_match is True


# --- evaluate_tool_calls -----------------------------------------------------


def test_evaluate_tool_calls_aggregates():
    entries_and_results = [
        # Full match
        (
            {"id": "t1", "kind": "tool", "expected_tool": "a", "expected_tool_args": {"x": 1}},
            _make_agent_result([ToolCall(tool_name="a", args={"x": 1}, result={})]),
        ),
        # Tool choice right, args wrong
        (
            {
                "id": "t2",
                "kind": "tool",
                "expected_tool": "a",
                "expected_tool_args": {"x": 1, "y": 2},
            },
            _make_agent_result([ToolCall(tool_name="a", args={"x": 1}, result={})]),
        ),
        # Wrong tool
        (
            {"id": "t3", "kind": "tool", "expected_tool": "a", "expected_tool_args": {"x": 1}},
            _make_agent_result([ToolCall(tool_name="b", args={"x": 1}, result={})]),
        ),
    ]
    report = evaluate_tool_calls(entries_and_results)
    assert report.n_evaluated == 3
    assert report.tool_choice_accuracy == pytest.approx(2 / 3)
    assert report.mean_args_match == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert report.full_match_rate == pytest.approx(1 / 3)


def test_evaluate_tool_calls_skips_qa_entries():
    """Q&A entries shouldn't be processed by tool eval."""
    pairs = [
        ({"id": "q1", "kind": "qa"}, _make_agent_result([])),
        (
            {"id": "t1", "kind": "tool", "expected_tool": "a", "expected_tool_args": {"x": 1}},
            _make_agent_result([ToolCall(tool_name="a", args={"x": 1}, result={})]),
        ),
    ]
    report = evaluate_tool_calls(pairs)
    assert report.n_evaluated == 1
    assert report.per_question[0].question_id == "t1"


def test_evaluate_tool_calls_empty_returns_zero_report():
    report = evaluate_tool_calls([])
    assert report.n_evaluated == 0
    assert report.per_question == []
    assert report.tool_choice_accuracy == 0.0
