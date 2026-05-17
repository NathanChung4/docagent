"""Tests for the LLM-as-judge answer evaluator.

Mocks `client.messages.create(...)` since the judge uses non-streaming calls
(unlike Generator/Agent which stream). All tests run offline — no API key.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from knowledge_rag.answer_eval import (
    DEFAULT_PASS_THRESHOLD,
    JudgeVerdict,
    _local_hits_misses,
    _parse_judge_response,
    evaluate_answers,
    judge_answer,
)

# --- Fakes -------------------------------------------------------------------


class _FakeJudgeMessages:
    """Fake `client.messages` that returns scripted JSON responses to .create()."""

    def __init__(self, scripted_texts: Sequence[str], usages: Sequence[Any] | None = None) -> None:
        self._texts = list(scripted_texts)
        self._usages = (
            list(usages)
            if usages is not None
            else [SimpleNamespace(input_tokens=120, output_tokens=20) for _ in scripted_texts]
        )
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._texts:
            raise AssertionError("Judge issued more calls than the script provides.")
        text = self._texts.pop(0)
        usage = self._usages.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=usage,
        )


class _FakeJudgeClient:
    def __init__(self, scripted_texts: Sequence[str], usages: Sequence[Any] | None = None) -> None:
        self.messages = _FakeJudgeMessages(scripted_texts, usages)


# --- Local helpers -----------------------------------------------------------


def test_local_hits_misses_case_insensitive_substring():
    hits, misses = _local_hits_misses(["AAA", "bbb", "ccc"], "result was AAA and BBB only")
    assert hits == ["AAA", "bbb"]
    assert misses == ["ccc"]


def test_local_hits_misses_empty_expected():
    assert _local_hits_misses([], "anything") == ([], [])


# --- Parser ------------------------------------------------------------------


def test_parse_judge_response_strict_json():
    score, rationale = _parse_judge_response('{"score": 0.85, "rationale": "good answer"}')
    assert score == 0.85
    assert rationale == "good answer"


def test_parse_judge_response_strips_code_fences():
    raw = '```json\n{"score": 1.0, "rationale": "perfect"}\n```'
    score, rationale = _parse_judge_response(raw)
    assert score == 1.0
    assert rationale == "perfect"


def test_parse_judge_response_handles_preamble():
    raw = 'Sure, here is my grading: {"score": 0.5, "rationale": "partial"}'
    score, rationale = _parse_judge_response(raw)
    assert score == 0.5
    assert rationale == "partial"


def test_parse_judge_response_rejects_no_json():
    with pytest.raises(ValueError):
        _parse_judge_response("totally not json")


def test_parse_judge_response_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        _parse_judge_response('{"score": 2.5, "rationale": "..."}')


# --- judge_answer ------------------------------------------------------------


def test_judge_answer_passing_score():
    client = _FakeJudgeClient(['{"score": 0.9, "rationale": "matches all fragments"}'])
    verdict = judge_answer(
        question="What is X?",
        expected_contains=["X is 5"],
        generated_answer="X is 5 according to the docs.",
        client=client,
        question_id="q1",
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.9
    assert verdict.passed is True
    assert verdict.expected_contains_hits == ["X is 5"]
    assert verdict.expected_contains_misses == []
    assert verdict.judge_input_tokens == 120
    assert verdict.judge_output_tokens == 20
    assert verdict.judge_cost_usd > 0
    assert verdict.parse_error is False


def test_judge_answer_failing_below_threshold():
    client = _FakeJudgeClient(['{"score": 0.4, "rationale": "missing key fragment"}'])
    verdict = judge_answer(
        question="What is X?",
        expected_contains=["X is 5"],
        generated_answer="X is something.",
        client=client,
        pass_threshold=0.7,
    )
    assert verdict.score == 0.4
    assert verdict.passed is False
    assert verdict.expected_contains_misses == ["X is 5"]


def test_judge_answer_idk_case_passes_when_disclaimed():
    client = _FakeJudgeClient(['{"score": 1.0, "rationale": "disclaimed correctly"}'])
    verdict = judge_answer(
        question="What is the price?",
        expected_contains=[],
        generated_answer="I don't know based on the available documents.",
        client=client,
    )
    assert verdict.score == 1.0
    assert verdict.passed is True


def test_judge_answer_retries_on_parse_failure():
    """First response is non-JSON; judge gets one retry with a stricter prompt."""
    client = _FakeJudgeClient(
        [
            "I think this answer is good!",
            '{"score": 0.8, "rationale": "good"}',
        ],
    )
    verdict = judge_answer(
        question="What is X?",
        expected_contains=["X is 5"],
        generated_answer="X is 5.",
        client=client,
    )
    assert verdict.score == 0.8
    assert verdict.passed is True
    assert verdict.parse_error is False
    # Tokens should sum across both calls.
    assert verdict.judge_input_tokens == 240
    assert verdict.judge_output_tokens == 40
    # Confirm the retry prompt was actually sent.
    assert len(client.messages.calls) == 2
    assert "JSON object on a single line" in client.messages.calls[1]["messages"][0]["content"]


def test_judge_answer_records_parse_error_after_two_failures():
    client = _FakeJudgeClient(["nope", "still nope"])
    verdict = judge_answer(
        question="What is X?",
        expected_contains=["X is 5"],
        generated_answer="X is 5.",
        client=client,
    )
    assert verdict.score == 0.0
    assert verdict.passed is False
    assert verdict.parse_error is True
    assert "parse_error" in verdict.rationale


def test_judge_answer_prices_by_model_name():
    """Switching --judge-model should swap pricing too — Sonnet costs 3x Haiku per input token."""
    haiku_client = _FakeJudgeClient(['{"score": 0.9, "rationale": "x"}'])
    sonnet_client = _FakeJudgeClient(['{"score": 0.9, "rationale": "x"}'])
    haiku_v = judge_answer(
        question="q",
        expected_contains=["a"],
        generated_answer="a",
        client=haiku_client,
        model="claude-haiku-4-5",
    )
    sonnet_v = judge_answer(
        question="q",
        expected_contains=["a"],
        generated_answer="a",
        client=sonnet_client,
        model="claude-sonnet-4-6",
    )
    assert sonnet_v.judge_cost_usd == pytest.approx(3 * haiku_v.judge_cost_usd, rel=1e-9)


def test_judge_answer_uses_default_threshold():
    """Default threshold is exposed; verdict.passed reflects it."""
    client = _FakeJudgeClient([f'{{"score": {DEFAULT_PASS_THRESHOLD}, "rationale": "edge"}}'])
    verdict = judge_answer(
        question="Q",
        expected_contains=["x"],
        generated_answer="x",
        client=client,
    )
    # Default threshold uses >= so equal passes.
    assert verdict.passed is True


# --- evaluate_answers --------------------------------------------------------


def test_evaluate_answers_aggregates_correctly():
    client = _FakeJudgeClient(
        [
            '{"score": 1.0, "rationale": "perfect"}',
            '{"score": 0.5, "rationale": "partial"}',
            '{"score": 0.9, "rationale": "good"}',
        ],
    )
    pairs = [
        (
            {"id": "q1", "question": "Q1", "expected_answer_contains": ["a"], "kind": "qa"},
            "answer a",
        ),
        (
            {"id": "q2", "question": "Q2", "expected_answer_contains": ["b"], "kind": "qa"},
            "answer b-ish",
        ),
        (
            {"id": "q3", "question": "Q3", "expected_answer_contains": ["c"], "kind": "qa"},
            "answer c",
        ),
    ]
    report = evaluate_answers(pairs, client=client, pass_threshold=0.7)
    assert len(report.per_question) == 3
    # q1 (1.0) and q3 (0.9) pass at 0.7; q2 (0.5) fails.
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.mean_score == pytest.approx((1.0 + 0.5 + 0.9) / 3)
    assert report.total_cost_usd > 0
    assert report.n_parse_errors == 0


def test_evaluate_answers_skips_non_qa_kinds():
    """Tool entries shouldn't be graded by the answer judge."""
    client = _FakeJudgeClient(['{"score": 0.9, "rationale": "good"}'])
    pairs = [
        ({"id": "q1", "question": "Q1", "expected_answer_contains": ["a"], "kind": "qa"}, "a"),
        ({"id": "q2", "question": "Q2", "kind": "tool", "expected_tool": "foo"}, ""),
    ]
    report = evaluate_answers(pairs, client=client)
    # Only the qa entry should have been judged.
    assert len(report.per_question) == 1
    assert report.per_question[0].question_id == "q1"
    assert len(client.messages.calls) == 1


def test_evaluate_answers_empty_pairs():
    client = _FakeJudgeClient([])
    report = evaluate_answers([], client=client)
    assert report.per_question == []
    assert report.pass_rate == 0.0
    assert report.mean_score == 0.0
    assert report.total_cost_usd == 0.0
