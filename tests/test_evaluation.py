"""Unit tests for the eval metrics.

These verify the metric *implementations* against known inputs — if these are
wrong, every baseline number we record is meaningless. Kept independent of the
real retriever (uses a fake) so they're fast and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from knowledge_rag.evaluation import (
    AggregateMetrics,
    EvalReport,
    aggregate,
    evaluate_question,
    evaluate_retriever,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowledge_rag.models import Chunk, Document, RetrievalResult, SourceType


def _make_result(uri: str, title: str = "", score: float = 1.0) -> RetrievalResult:
    """Build a RetrievalResult with just enough Chunk to satisfy the matcher."""
    doc = Document(source_type=SourceType.WIKI, title=title or uri, content="", uri=uri)
    chunk = Chunk.from_document(doc, content="")
    return RetrievalResult(chunk=chunk, score=score)


# --- recall_at_k -------------------------------------------------------------


class TestRecallAtK:
    def test_all_expected_sources_in_top_k(self):
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/data/sample/confluence/pressure_sensor.html"),
        ]
        assert recall_at_k(results, ["flow_controller.html", "pressure_sensor.html"], k=5) == 1.0

    def test_partial_match(self):
        results = [_make_result("/data/sample/confluence/flow_controller.html")]
        # 1 of 2 expected sources matched
        assert recall_at_k(results, ["flow_controller.html", "pressure_sensor.html"], k=5) == 0.5

    def test_none_matched(self):
        results = [_make_result("/data/sample/confluence/unrelated.html")]
        assert recall_at_k(results, ["flow_controller.html"], k=5) == 0.0

    def test_match_outside_top_k_doesnt_count(self):
        results = [
            _make_result("/x.html"),
            _make_result("/y.html"),
            _make_result("/data/sample/confluence/flow_controller.html"),
        ]
        # k=2 → the relevant result at position 3 is missed
        assert recall_at_k(results, ["flow_controller.html"], k=2) == 0.0
        # k=3 → it's now included
        assert recall_at_k(results, ["flow_controller.html"], k=3) == 1.0

    def test_empty_expected_sources_returns_zero(self):
        results = [_make_result("/x.html")]
        assert recall_at_k(results, [], k=5) == 0.0

    def test_empty_results(self):
        assert recall_at_k([], ["flow_controller.html"], k=5) == 0.0

    def test_match_via_title_when_uri_doesnt_match(self):
        results = [_make_result(uri="/abs/path/123.html", title="flow_controller")]
        assert recall_at_k(results, ["flow_controller"], k=5) == 1.0

    def test_case_insensitive(self):
        results = [_make_result("/data/sample/confluence/Flow_Controller.HTML")]
        assert recall_at_k(results, ["flow_controller.html"], k=5) == 1.0


# --- precision_at_k ----------------------------------------------------------


class TestPrecisionAtK:
    def test_all_top_k_relevant(self):
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/data/sample/confluence/pressure_sensor.html"),
        ]
        assert precision_at_k(results, ["flow_controller.html", "pressure_sensor.html"], k=2) == 1.0

    def test_two_of_five_relevant(self):
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/x.html"),
            _make_result("/y.html"),
            _make_result("/data/sample/confluence/pressure_sensor.html"),
            _make_result("/z.html"),
        ]
        assert precision_at_k(results, ["flow_controller.html", "pressure_sensor.html"], k=5) == 0.4

    def test_denominator_is_actual_top_k_when_results_short(self):
        # k=5 requested but only 2 results returned — denominator should be 2,
        # not 5, so the retriever isn't punished for empty slots.
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/x.html"),
        ]
        assert precision_at_k(results, ["flow_controller.html"], k=5) == 0.5

    def test_empty_results(self):
        assert precision_at_k([], ["flow_controller.html"], k=5) == 0.0


# --- reciprocal_rank ---------------------------------------------------------


class TestReciprocalRank:
    def test_first_position(self):
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/x.html"),
        ]
        assert reciprocal_rank(results, ["flow_controller.html"]) == 1.0

    def test_third_position(self):
        results = [
            _make_result("/x.html"),
            _make_result("/y.html"),
            _make_result("/data/sample/confluence/flow_controller.html"),
        ]
        assert reciprocal_rank(results, ["flow_controller.html"]) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self):
        results = [_make_result("/x.html"), _make_result("/y.html")]
        assert reciprocal_rank(results, ["flow_controller.html"]) == 0.0

    def test_only_first_match_counts(self):
        # Even if the second result is also relevant, RR uses only the first.
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/data/sample/confluence/pressure_sensor.html"),
        ]
        rr = reciprocal_rank(results, ["flow_controller.html", "pressure_sensor.html"])
        assert rr == 1.0


# --- evaluate_question -------------------------------------------------------


class TestEvaluateQuestion:
    def test_records_metrics_at_each_k(self):
        results = [
            _make_result("/data/sample/confluence/flow_controller.html"),
            _make_result("/x.html"),
            _make_result("/y.html"),
        ]
        qr = evaluate_question(
            question_id="q1",
            question="What is X?",
            results=results,
            expected_sources=["flow_controller.html"],
            k_values=(1, 5),
        )
        assert qr.recall_at_k == {1: 1.0, 5: 1.0}
        assert qr.precision_at_k[1] == 1.0
        assert qr.precision_at_k[5] == pytest.approx(1 / 3)
        assert qr.reciprocal_rank == 1.0
        assert qr.retrieved_sources == [r.chunk.uri for r in results]
        assert qr.expected_sources == ["flow_controller.html"]


# --- aggregate ---------------------------------------------------------------


class TestAggregate:
    def test_macro_average(self):
        # Three questions with different per-question scores; aggregate is the
        # plain mean (not weighted by expected-source count).
        per_q = [
            evaluate_question(
                "q1", "?", [_make_result("/match1.html")], ["match1.html"], k_values=(5,)
            ),
            evaluate_question(
                "q2",
                "?",
                [_make_result("/wrong.html")],
                ["match2.html"],
                k_values=(5,),
            ),
            evaluate_question(
                "q3",
                "?",
                [_make_result("/wrong.html"), _make_result("/match3.html")],
                ["match3.html"],
                k_values=(5,),
            ),
        ]
        agg = aggregate(per_q, k_values=(5,))
        assert agg.n_questions == 3
        # Recall: 1.0 + 0.0 + 1.0 → mean 0.667
        assert agg.mean_recall_at_k[5] == pytest.approx(2 / 3)
        # MRR: 1.0 + 0.0 + 0.5 → mean 0.5
        assert agg.mrr == pytest.approx(0.5)

    def test_empty(self):
        agg = aggregate([], k_values=(5, 10))
        assert agg.n_questions == 0
        assert agg.mrr == 0.0
        assert agg.mean_recall_at_k == {5: 0.0, 10: 0.0}


# --- evaluate_retriever (integration with fake) ------------------------------


@dataclass
class _FakeRetriever:
    """Returns a canned list of results regardless of query.

    Lets us exercise evaluate_retriever's loop + skip behavior without spinning
    up the real embedder + ChromaDB.
    """

    canned: list[RetrievalResult]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        alpha: float | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievalResult]:
        return self.canned[:k]


class TestEvaluateRetriever:
    def test_skips_tool_kind_entries(self):
        retriever = _FakeRetriever(canned=[_make_result("/x.html")])
        dataset = [
            {
                "id": "q1",
                "question": "?",
                "expected_sources": ["x.html"],
                "kind": "qa",
            },
            {"id": "q2", "question": "?", "expected_tool": "foo", "kind": "tool"},
        ]
        report = evaluate_retriever(retriever, dataset, k_values=(5,))
        assert report.aggregate.n_questions == 1
        assert any(s["id"] == "q2" for s in report.skipped)

    def test_skips_entries_without_expected_sources(self):
        retriever = _FakeRetriever(canned=[_make_result("/x.html")])
        dataset = [
            {"id": "q1", "question": "?", "expected_sources": [], "kind": "qa"},
        ]
        report = evaluate_retriever(retriever, dataset, k_values=(5,))
        assert report.aggregate.n_questions == 0
        assert report.skipped == [{"id": "q1", "reason": "no expected_sources"}]

    def test_defaults_kind_to_qa_when_missing(self):
        retriever = _FakeRetriever(canned=[_make_result("/match.html")])
        dataset = [
            {"id": "q1", "question": "?", "expected_sources": ["match.html"]},
        ]
        report = evaluate_retriever(retriever, dataset, k_values=(5,))
        assert report.aggregate.n_questions == 1
        assert report.aggregate.mean_recall_at_k[5] == 1.0

    def test_per_question_results_populated(self):
        retriever = _FakeRetriever(
            canned=[
                _make_result("/match.html"),
                _make_result("/junk.html"),
            ]
        )
        dataset = [
            {
                "id": "q1",
                "question": "Find the match",
                "expected_sources": ["match.html"],
                "kind": "qa",
            },
        ]
        report = evaluate_retriever(retriever, dataset, k_values=(5, 10))
        assert len(report.per_question) == 1
        qr = report.per_question[0]
        assert qr.question_id == "q1"
        assert qr.reciprocal_rank == 1.0
        assert qr.recall_at_k == {5: 1.0, 10: 1.0}

    def test_returns_eval_report(self):
        retriever = _FakeRetriever(canned=[])
        report = evaluate_retriever(retriever, [], k_values=(5,))
        assert isinstance(report, EvalReport)
        assert isinstance(report.aggregate, AggregateMetrics)
