"""Tests for CrossEncoderReranker.

The real reranker downloads ~280MB of model weights on first call. We split:
    - fast tests use a stub model so import / wiring / mutation contract is
      verified in CI without the download
    - slow tests load the real model and check that an obviously-better chunk
      gets promoted; marked @pytest.mark.slow so they're easy to skip

Run only fast tests: `pytest tests/test_reranker.py -m 'not slow'`.
Run everything (downloads the model): `pytest tests/test_reranker.py`.
"""

from __future__ import annotations

import pytest

from knowledge_rag.models import Chunk, RetrievalResult, SourceType
from knowledge_rag.reranker import CrossEncoderReranker


def _result(chunk_id: str, content: str, score: float = 0.5) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        content=content,
        source_type=SourceType.WIKI,
        title="t",
        uri="u",
    )
    return RetrievalResult(chunk=chunk, score=score)


class _StubModel:
    """Mimic CrossEncoder.predict — return a fixed score per (query, chunk) pair."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, show_progress_bar: bool = False):
        self.calls.append(list(pairs))
        return self.scores


def test_rerank_empty_results_returns_empty() -> None:
    rr = CrossEncoderReranker()
    assert rr.rerank("query", [], top_n=5) == []


def test_rerank_empty_query_returns_truncated_passthrough() -> None:
    rr = CrossEncoderReranker()
    results = [_result("a", "x"), _result("b", "y"), _result("c", "z")]
    out = rr.rerank("", results, top_n=2)
    assert len(out) == 2
    # No model call, no rerank_score writes.
    assert all(r.rerank_score is None for r in out)


def test_rerank_writes_score_and_reorders() -> None:
    """With a stub model, verify reorder by descending rerank_score."""
    rr = CrossEncoderReranker()
    rr._model = _StubModel(scores=[0.1, 0.9, 0.5])  # b > c > a
    results = [_result("a", "x"), _result("b", "y"), _result("c", "z")]
    out = rr.rerank("any query", results, top_n=3)
    assert [r.chunk.chunk_id for r in out] == ["b", "c", "a"]
    assert out[0].rerank_score == 0.9
    assert out[1].rerank_score == 0.5
    assert out[2].rerank_score == pytest.approx(0.1)


def test_rerank_truncates_to_top_n() -> None:
    rr = CrossEncoderReranker()
    rr._model = _StubModel(scores=[0.1, 0.9, 0.5, 0.7])
    results = [_result(c, c) for c in ["a", "b", "c", "d"]]
    out = rr.rerank("query", results, top_n=2)
    assert len(out) == 2
    assert [r.chunk.chunk_id for r in out] == ["b", "d"]


def test_rerank_pairs_query_with_each_chunk_content() -> None:
    rr = CrossEncoderReranker()
    stub = _StubModel(scores=[0.0, 0.0])
    rr._model = stub
    results = [_result("a", "alpha content"), _result("b", "beta content")]
    rr.rerank("my question", results, top_n=2)
    assert stub.calls == [[("my question", "alpha content"), ("my question", "beta content")]]


@pytest.mark.slow
def test_rerank_real_model_promotes_obvious_winner() -> None:
    """End-to-end with the real cross-encoder.

    Downloads BAAI/bge-reranker-base on first run. The 'obvious winner' chunk
    is a near-direct paraphrase of the query; a competent cross-encoder must
    rank it first.
    """
    rr = CrossEncoderReranker()
    query = "How do I configure the memory controller frequency?"
    results = [
        _result("cookies", "Cookies should be baked at 350 degrees for twelve minutes."),
        _result(
            "winner", "To set the memory controller's operating frequency, edit the config file."
        ),
        _result("widget", "The widget supports three power modes: low, medium, and high."),
    ]
    out = rr.rerank(query, results, top_n=3)
    assert out[0].chunk.chunk_id == "winner"
