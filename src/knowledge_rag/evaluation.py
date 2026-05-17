"""Retrieval-quality metrics for the RAG eval suite.

Three metrics, computed against a labeled question set:

  - **recall@k**: of the chunks that *should* have been retrieved, what
    fraction landed in the top-k? Answers "did we even find the right
    material?". The honest headline number for RAG when each question has only
    1–3 expected sources.

  - **precision@k**: of the k chunks returned, what fraction were relevant?
    Mostly useful for A/B comparisons (rerank on vs off) where the gold set
    is fixed but the top-k composition changes.

  - **MRR (Mean Reciprocal Rank)**: 1 / rank of the first relevant result,
    averaged across questions. Rank-sensitive in a way recall@k isn't —
    "right chunk at #1" beats "right chunk at #5" under MRR but ties under
    recall@5.

Dataset schema (per `Domain.eval_dataset()`):

    {
      "id": "q1",
      "question": "...",
      "expected_answer_contains": [...],   # used in Phase 9 answer eval, not here
      "expected_sources": ["flow_controller.html"],
      "kind": "qa"                         # or "tool" — tool entries are skipped
    }

Source matching is loose substring (case-insensitive) against each retrieved
chunk's URI or title. So `"flow_controller.html"` matches a chunk whose URI is
`/data/sample/confluence/flow_controller.html`. This is intentional — it lets
labellers write filenames without committing to absolute paths, and lets us
move data around without breaking labels. The trade-off is that a label like
`"flow"` would over-match; pick distinctive substrings.

This module is generic — zero domain-specific code. Phase 9 layers answer
relevance + tool-call accuracy on top.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from knowledge_rag.models import RetrievalResult

DEFAULT_K_VALUES: tuple[int, ...] = (5, 10)


class RetrieverProtocol(Protocol):
    """Minimal contract `evaluate_retriever` needs from a retriever.

    Stated as a Protocol so the eval module doesn't import the concrete
    `Retriever` class — keeps it usable with fakes (in tests) and with future
    retrievers that don't inherit from the same base.
    """

    def retrieve(self, query: str, k: int = ..., **kwargs: Any) -> list[RetrievalResult]: ...


# --- Per-question matching ---------------------------------------------------


def _chunk_matches_source(result: RetrievalResult, expected_source: str) -> bool:
    """A retrieved chunk satisfies an expected source if the source string
    appears (case-insensitive substring) in either the chunk's URI or title.
    """
    needle = expected_source.lower()
    return needle in result.chunk.uri.lower() or needle in result.chunk.title.lower()


def recall_at_k(
    results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Fraction of expected sources matched by at least one of the top-k results.

    Returns 0.0 if `expected_sources` is empty (caller should usually filter
    those questions out — they aren't measurable as retrieval problems).
    """
    if not expected_sources or k <= 0:
        return 0.0
    top_k = results[:k]
    matched = sum(
        1 for src in expected_sources if any(_chunk_matches_source(r, src) for r in top_k)
    )
    return matched / len(expected_sources)


def precision_at_k(
    results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Fraction of top-k results that match at least one expected source.

    Denominator is `min(k, len(results))` so a retriever returning fewer than
    k results isn't punished for the empty slots. Returns 0.0 if `results` is
    empty or k <= 0.
    """
    if not expected_sources or k <= 0 or not results:
        return 0.0
    top_k = results[:k]
    relevant = sum(
        1 for r in top_k if any(_chunk_matches_source(r, src) for src in expected_sources)
    )
    return relevant / len(top_k)


def reciprocal_rank(
    results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
) -> float:
    """1 / rank of the first relevant result. 0.0 if no relevant result is found."""
    if not expected_sources:
        return 0.0
    for i, r in enumerate(results, start=1):
        if any(_chunk_matches_source(r, src) for src in expected_sources):
            return 1.0 / i
    return 0.0


# --- Per-question + aggregate dataclasses ------------------------------------


@dataclass
class QuestionResult:
    """Metrics for a single question, plus the raw retrieval trace.

    `retrieved_sources` is captured so failure-mode bucketing (Phase 4.5 final
    step) can inspect what the retriever actually returned without re-running.
    """

    question_id: str
    question: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    reciprocal_rank: float


@dataclass
class AggregateMetrics:
    """Macro-averaged metrics across all evaluated questions.

    Macro (mean of per-question scores) rather than micro (pool everything
    first) — keeps each question equally weighted regardless of how many
    expected sources it has.
    """

    n_questions: int
    mean_recall_at_k: dict[int, float]
    mean_precision_at_k: dict[int, float]
    mrr: float


@dataclass
class EvalReport:
    """The full output of one eval run.

    The runner script wraps this with date + config snapshot before serializing.
    """

    per_question: list[QuestionResult]
    aggregate: AggregateMetrics
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    skipped: list[dict[str, Any]] = field(default_factory=list)


# --- Driver ------------------------------------------------------------------


def evaluate_question(
    question_id: str,
    question: str,
    results: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> QuestionResult:
    """Compute all metrics for one question against one retrieval result list."""
    return QuestionResult(
        question_id=question_id,
        question=question,
        expected_sources=list(expected_sources),
        retrieved_sources=[r.chunk.uri for r in results],
        recall_at_k={k: recall_at_k(results, expected_sources, k) for k in k_values},
        precision_at_k={k: precision_at_k(results, expected_sources, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(results, expected_sources),
    )


def aggregate(
    per_question: Sequence[QuestionResult],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> AggregateMetrics:
    """Macro-average per-question metrics into a single AggregateMetrics."""
    n = len(per_question)
    if n == 0:
        return AggregateMetrics(
            n_questions=0,
            mean_recall_at_k={k: 0.0 for k in k_values},
            mean_precision_at_k={k: 0.0 for k in k_values},
            mrr=0.0,
        )
    return AggregateMetrics(
        n_questions=n,
        mean_recall_at_k={k: sum(r.recall_at_k[k] for r in per_question) / n for k in k_values},
        mean_precision_at_k={
            k: sum(r.precision_at_k[k] for r in per_question) / n for k in k_values
        },
        mrr=sum(r.reciprocal_rank for r in per_question) / n,
    )


def evaluate_retriever(
    retriever: RetrieverProtocol,
    dataset: Sequence[dict[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> EvalReport:
    """Run the full eval suite against a Retriever.

    Skips entries whose `kind` is not `"qa"` (tool-use eval is Phase 9) and
    entries with no `expected_sources` (not measurable as retrieval). Skipped
    entries are recorded in the report so labellers can audit them.

    For the "rerank on vs off" A/B comparison, build the retriever with or
    without a reranker — the eval module doesn't gate reranking itself.

    Args:
        retriever: Anything implementing the `RetrieverProtocol`.
        dataset: Output of `Domain.eval_dataset()`.
        k_values: The k's to compute recall/precision at.

    Returns:
        An EvalReport with per-question results and macro-averaged aggregates.
    """
    fetch_k = max(k_values)
    per_question: list[QuestionResult] = []
    skipped: list[dict[str, Any]] = []

    for entry in dataset:
        kind = entry.get("kind", "qa")
        if kind != "qa":
            skipped.append({"id": entry.get("id"), "reason": f"kind={kind}"})
            continue
        expected = entry.get("expected_sources") or []
        if not expected:
            skipped.append({"id": entry.get("id"), "reason": "no expected_sources"})
            continue

        results = retriever.retrieve(entry["question"], k=fetch_k)
        per_question.append(
            evaluate_question(
                question_id=entry.get("id", ""),
                question=entry["question"],
                results=results,
                expected_sources=expected,
                k_values=k_values,
            )
        )

    return EvalReport(
        per_question=per_question,
        aggregate=aggregate(per_question, k_values=k_values),
        k_values=tuple(k_values),
        skipped=skipped,
    )
