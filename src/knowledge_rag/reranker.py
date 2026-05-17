"""Cross-encoder reranker — second-pass precision booster.

A bi-encoder (sentence-transformers) embeds query and chunk independently and
compares their vectors. Fast, but the comparison happens in a low-dimensional
space that throws away token-level interaction.

A cross-encoder takes `(query, chunk)` as a single pair, runs both through the
transformer together, and outputs one relevance score. The cross-attention sees
exact word overlap, syntactic structure, and contextual match — much more
accurate per pair, ~50ms per pair on CPU.

The economics: too slow to run on the full corpus, fast enough to rescore a
candidate pool of ~20. So the pattern is: hybrid retrieval narrows to top-20
cheaply, the cross-encoder rescores those 20 → keep top-5.

Default model: BAAI/bge-reranker-base — open-weights, ~280MB, strong English
retrieval baseline. Score is a logit (unbounded, sign indicates relevance).
"""

from __future__ import annotations

from collections.abc import Sequence

from knowledge_rag.models import RetrievalResult

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Wrap a sentence-transformers CrossEncoder for query/chunk reranking.

    The model is lazy-loaded on first `rerank()` call so importing this module
    is cheap and tests that don't actually rerank stay fast.

    Attributes:
        model_name: Hugging Face id of the cross-encoder model.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None  # type: ignore[var-annotated]

    def _load(self):
        """Lazy-load the CrossEncoder (downloads weights on first call)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        """Rescore `results` against `query` with the cross-encoder.

        Mutates the input results' `rerank_score` field in place (each result
        is the same object the caller passed in), then returns a new list
        sorted by rerank_score descending and truncated to `top_n`.

        Args:
            query: User query string.
            results: Candidate hits from the hybrid retriever.
            top_n: Number of reranked results to return.
        """
        if not results or not query:
            return list(results[:top_n])

        model = self._load()
        pairs = [(query, r.chunk.content) for r in results]
        # CrossEncoder.predict returns a numpy array of logits — higher = more relevant.
        scores = model.predict(pairs, show_progress_bar=False)

        for result, score in zip(results, scores, strict=True):
            result.rerank_score = float(score)

        ordered = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
        return list(ordered[:top_n])
