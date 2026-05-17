"""Tool-call accuracy evaluation for the agent loop.

Pairs with `evaluation.py` (retrieval) and `answer_eval.py` (answer quality):
together they cover all three observable behaviors of the agentic RAG system.

The metric this module computes is "did the agent pick the right tool with
roughly the right arguments?" — independent of whether the tool itself
succeeded. A tool can be picked correctly and still fail (missing data file,
out-of-range value), and we want both signals separately. The eval entries
distinguish them via two flags:

  - `expected_tool` — the tool that should be invoked. `None` means "no tool
    should be called" (Q&A-style prompt that shouldn't trigger one).
  - `expected_validation_error` — `True` means the agent SHOULD trigger a
    ToolValidationError when it dispatches the tool, demonstrating that the
    feedback loop from Phase 6 is working.

Args matching is intentionally lenient — the agent paraphrases user prompts
into tool args, so an exact JSON-equal match is too strict. Strategy:

  1. Required keys (those listed in expected_args) must all be present.
  2. Each missing required key drops the score by 1/N (N = #required keys).
  3. Each extra key the model invented penalizes 0.1 (capped at 1.0 lost).
  4. Value match: exact for ints/strings/bools; `math.isclose(rel_tol=1e-3)`
     for floats; recursive for nested dicts (e.g., `params`).

A `full_match` requires tool_choice_correct AND args_match_score == 1.0.

Generic — zero domain-specific tool names.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from knowledge_rag.agent import AgentResult
from knowledge_rag.models import ToolCall


@dataclass
class ToolEvalVerdict:
    """Per-question tool eval result. Aggregated into ToolEvalReport."""

    question_id: str
    question: str
    expected_tool: str | None
    expected_args: dict[str, Any] | None
    actual_tools_called: list[str]
    tool_choice_correct: bool
    args_match_score: float
    full_match: bool
    expected_validation_error: bool
    actual_validation_error: bool
    validation_error_correct: bool


@dataclass
class ToolEvalReport:
    """Aggregate output across the dataset's tool entries."""

    per_question: list[ToolEvalVerdict] = field(default_factory=list)
    n_evaluated: int = 0
    tool_choice_accuracy: float = 0.0
    mean_args_match: float = 0.0
    full_match_rate: float = 0.0
    validation_error_accuracy: float = 0.0


# --- args matching -----------------------------------------------------------


def _values_match(expected: Any, actual: Any) -> bool:
    """Lenient value comparison: exact for most types, isclose for floats, recursive for dicts."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _dict_args_match(expected, actual) == 1.0
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-3, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_values_match(e, a) for e, a in zip(expected, actual, strict=True))
    return expected == actual


def _dict_args_match(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Score 0.0–1.0 for how well `actual` covers `expected`.

    Required-key recall first: each missing required key drops 1/N from the
    score. Extra keys penalize 0.1 each (capped at 1.0 total).
    """
    if not expected:
        # Nothing required; score by absence of extras.
        extra_keys = len(actual)
        return max(0.0, 1.0 - 0.1 * extra_keys)

    n_required = len(expected)
    matched = 0
    for key, exp_val in expected.items():
        if key in actual and _values_match(exp_val, actual[key]):
            matched += 1
    base_score = matched / n_required

    extra_keys = sum(1 for k in actual if k not in expected)
    penalty = min(1.0, 0.1 * extra_keys)
    return max(0.0, base_score - penalty)


def _has_validation_error(tool_calls: Sequence[ToolCall], expected_tool: str | None) -> bool:
    """True iff at least one tool dispatch surfaced a ToolValidationError.

    When `expected_tool` is set, only counts errors on calls to that tool —
    avoids false positives from secondary tool dispatches the agent may try.
    """
    for tc in tool_calls:
        if tc.success:
            continue
        if expected_tool is None or tc.tool_name == expected_tool:
            return True
    return False


# --- per-question verdict ----------------------------------------------------


def evaluate_tool_call(entry: dict, agent_result: AgentResult) -> ToolEvalVerdict:
    """Score one (eval entry, agent run) pair.

    The entry's schema (additions to the generic eval format):
        kind: "tool"
        expected_tool: str | None
        expected_tool_args: dict | None  (omit for negative-tool entries)
        expected_validation_error: bool  (default False)

    Returns a ToolEvalVerdict whether or not the agent called any tool — the
    "no tool when one was expected" case scores 0 cleanly, and the "tool when
    none was expected" case is tracked too.
    """
    expected_tool = entry.get("expected_tool")
    expected_args = entry.get("expected_tool_args")
    expected_val_err = bool(entry.get("expected_validation_error", False))

    tool_calls = agent_result.query_log.tool_calls
    actual_tools = [tc.tool_name for tc in tool_calls]

    # tool_choice_correct logic:
    #   expected None: correct iff agent called no tools
    #   expected set:  correct iff at least one call matched
    if expected_tool is None:
        tool_choice_correct = len(actual_tools) == 0
    else:
        tool_choice_correct = expected_tool in actual_tools

    # args match: only meaningful when a tool was expected AND the right one
    # was called. Otherwise score 0 (no args to score) or 1 (no args expected,
    # no tool called — degenerate but consistent).
    if expected_args is None:
        args_score = 1.0 if tool_choice_correct else 0.0
    elif not tool_choice_correct:
        args_score = 0.0
    else:
        # Score against the FIRST call to the expected tool. Multiple calls
        # to the same tool would be a separate signal worth tracking, but for
        # now first-call wins — keeps the scoring deterministic.
        first_match = next(tc for tc in tool_calls if tc.tool_name == expected_tool)
        args_score = _dict_args_match(expected_args, first_match.args)

    full_match = tool_choice_correct and args_score == 1.0

    actual_val_err = _has_validation_error(tool_calls, expected_tool)
    val_err_correct = actual_val_err == expected_val_err

    return ToolEvalVerdict(
        question_id=entry.get("id", ""),
        question=entry.get("question", ""),
        expected_tool=expected_tool,
        expected_args=expected_args,
        actual_tools_called=actual_tools,
        tool_choice_correct=tool_choice_correct,
        args_match_score=args_score,
        full_match=full_match,
        expected_validation_error=expected_val_err,
        actual_validation_error=actual_val_err,
        validation_error_correct=val_err_correct,
    )


def evaluate_tool_calls(
    pairs: Sequence[tuple[dict, AgentResult]],
) -> ToolEvalReport:
    """Score many (entry, agent_result) pairs. Skips entries whose `kind` isn't 'tool'.

    Aggregates: tool_choice_accuracy, mean_args_match, full_match_rate, and
    validation_error_accuracy (how often the agent's actual validation-error
    behavior matched the expected one — both for entries that should error
    and entries that shouldn't).
    """
    verdicts = [
        evaluate_tool_call(entry, agent_result)
        for entry, agent_result in pairs
        if entry.get("kind") == "tool"
    ]
    n = len(verdicts)
    if n == 0:
        return ToolEvalReport()

    return ToolEvalReport(
        per_question=verdicts,
        n_evaluated=n,
        tool_choice_accuracy=sum(1 for v in verdicts if v.tool_choice_correct) / n,
        mean_args_match=sum(v.args_match_score for v in verdicts) / n,
        full_match_rate=sum(1 for v in verdicts if v.full_match) / n,
        validation_error_accuracy=sum(1 for v in verdicts if v.validation_error_correct) / n,
    )
