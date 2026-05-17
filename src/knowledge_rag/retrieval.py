"""Hybrid retrieval pipeline: semantic + BM25 keyword search.

Why hybrid: dense embeddings catch paraphrase but blur exact tokens (flag names,
error codes, IDs). BM25 catches exact tokens but misses paraphrase. Combining
them recovers both classes of relevant chunks.

Components:
    - BM25Index: an in-memory keyword index over a snapshot of the corpus.
      Manual rebuild — caller invokes `rebuild(chunks)` after ingestion. The
      simplicity is intentional: at small/medium corpus scale, an in-process
      BM25 over a Python list is the right tool. Production scale would move
      keyword search to OpenSearch/Elasticsearch and replace this class.
    - HybridRetriever: composes a VectorStore (semantic) and a BM25Index.
      Min-max normalizes each ranked list, then blends with a configurable
      `alpha` ∈ [0, 1]. alpha=1 → semantic only; alpha=0 → BM25 only.
    - Retriever: end-to-end facade that wires hybrid retrieval and (optionally)
      a cross-encoder reranker behind one `retrieve()` method. This is the
      single entrypoint downstream callers (generation, agent) should use.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from knowledge_rag.models import Chunk, RetrievalResult
from knowledge_rag.reranker import CrossEncoderReranker
from knowledge_rag.vectorstore import VectorStore

DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 5
DEFAULT_ALPHA = 0.5

# BM25 doesn't need linguistic stemming — a plain word-token bag is the
# standard baseline and what our tests assume.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _matches_filter(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Equality-only subset of ChromaDB's `where` semantics.

    Anything fancier ($in, $gt, etc.) would require parsing — out of scope
    until a real call site needs it.
    """
    if not where:
        return True
    for key, expected in where.items():
        if metadata.get(key) != expected:
            return False
    return True


def _minmax(scores: list[float]) -> list[float]:
    """Scale `scores` into [0, 1] via min-max. All-equal input maps to all 1s."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        # Treat as all-relevant rather than all-zero so one retriever's
        # contribution isn't silently nulled out in the blend.
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


# --- BM25 index --------------------------------------------------------------


class BM25Index:
    """In-memory BM25 index over a snapshot of the corpus.

    Lifecycle: construct empty, call `rebuild(chunks)` once the corpus is
    known, then query. Re-call `rebuild` after the corpus changes. Rebuilt
    from scratch each time because BM25's IDF depends on the full corpus —
    incremental updates would require maintaining DF counts ourselves.
    """

    def __init__(self) -> None:
        self._bm25: Any = None  # rank_bm25.BM25Okapi (lazy-imported)
        self._chunks: list[Chunk] = []

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        """Replace the index with a fresh BM25 fit over `chunks`."""
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def query(
        self,
        text: str,
        k: int = DEFAULT_CANDIDATE_K,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return up to `k` highest-BM25 chunks matching the optional filter.

        Score is the raw BM25 sum (unbounded). HybridRetriever normalizes
        before blending across legs.
        """
        if not text or self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(text))
        scored = [
            (chunk, float(score))
            for chunk, score in zip(self._chunks, scores, strict=True)
            if _matches_filter(chunk.to_filter_dict(), where)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in scored[:k]]

    def __len__(self) -> int:
        return len(self._chunks)


# --- Hybrid retriever --------------------------------------------------------


@dataclass
class HybridConfig:
    """Tunable parameters for hybrid retrieval. Defaults are sensible pre-eval picks."""

    candidate_k: int = DEFAULT_CANDIDATE_K
    alpha: float = DEFAULT_ALPHA  # 1.0 = semantic only, 0.0 = BM25 only


class HybridRetriever:
    """Combine semantic (VectorStore) and keyword (BM25Index) retrieval.

    Strategy:
        1. Fetch top-`candidate_k` from each leg (over-fetch so the union has
           enough material).
        2. Min-max normalize each leg's scores into [0, 1].
        3. Blend `alpha * semantic_norm + (1 - alpha) * bm25_norm`. Missing
           in one leg → that leg contributes 0.
        4. Sort by blended score, return top-`k`.

    The `alpha` parameter is the headline tunable; defaults to 0.5 and gets
    tuned against the eval set in Phase 9.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        config: HybridConfig | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.config = config or HybridConfig()

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_FINAL_K,
        where: dict[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval. Returns top-`k` by blended score.

        Args:
            query: Natural-language query string.
            k: Number of results to return after blending.
            where: Optional metadata filter applied to both legs.
            alpha: Override the configured blend weight for this call.
        """
        if not query:
            return []

        a = self.config.alpha if alpha is None else alpha
        a = max(0.0, min(1.0, a))
        cand = self.config.candidate_k

        # Skip the leg whose contribution will be multiplied by zero. Saves the
        # query embedding (semantic) or the O(N) BM25 score sweep when the user
        # is doing single-leg ablations.
        semantic_hits = self.vector_store.query(query, k=cand, where=where) if a > 0.0 else []
        keyword_hits = self.bm25_index.query(query, k=cand, where=where) if a < 1.0 else []

        sem_norm = _minmax([h.score for h in semantic_hits])
        kw_norm = _minmax([h.score for h in keyword_hits])

        bag: dict[str, tuple[Chunk, float, float]] = {}
        for hit, s in zip(semantic_hits, sem_norm, strict=True):
            bag[hit.chunk.chunk_id] = (hit.chunk, s, 0.0)
        for hit, s in zip(keyword_hits, kw_norm, strict=True):
            existing = bag.get(hit.chunk.chunk_id)
            if existing is None:
                bag[hit.chunk.chunk_id] = (hit.chunk, 0.0, s)
            else:
                chunk, sem_s, _ = existing
                bag[hit.chunk.chunk_id] = (chunk, sem_s, s)

        blended = [
            RetrievalResult(chunk=chunk, score=a * sem_s + (1.0 - a) * kw_s)
            for chunk, sem_s, kw_s in bag.values()
        ]
        blended.sort(key=lambda r: r.score, reverse=True)
        return blended[:k]


# --- Retriever facade --------------------------------------------------------


class Retriever:
    """End-to-end retriever: hybrid + (optional) cross-encoder reranker.

    This is the entrypoint downstream code (generation, agent loop) should
    depend on. Owns the over-fetch → blend → rerank → cut pipeline so callers
    don't have to wire three components.

    The reranker is optional: pass `reranker=None` to skip the rerank step
    (useful for the Phase 9 A/B test comparing with vs without rerank).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: CrossEncoderReranker | None = None,
        hybrid_config: HybridConfig | None = None,
        candidate_k: int | None = None,
    ) -> None:
        self.hybrid = HybridRetriever(vector_store, bm25_index, hybrid_config)
        self.reranker = reranker
        # Owned by Retriever rather than reaching through to hybrid.config so
        # the facade can over-fetch independently of the hybrid leg's config.
        self.candidate_k = (
            candidate_k if candidate_k is not None else self.hybrid.config.candidate_k
        )

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_FINAL_K,
        where: dict[str, Any] | None = None,
        alpha: float | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievalResult]:
        """End-to-end retrieve: hybrid candidates → rerank → top-`k`.

        Args:
            query: User query string.
            k: Final result count.
            where: Metadata filter passed through to both legs.
            alpha: Override hybrid blend weight (None = use config).
            use_rerank: If False or no reranker is configured, skip rerank
                and return the top-k blended results directly.
        """
        will_rerank = use_rerank and self.reranker is not None
        fetch_k = self.candidate_k if will_rerank else k
        candidates = self.hybrid.retrieve(query, k=fetch_k, where=where, alpha=alpha)

        if will_rerank and candidates:
            return self.reranker.rerank(query, candidates, top_n=k)
        return candidates[:k]

    def rebuild_bm25(self, chunks: Sequence[Chunk]) -> None:
        """Refit BM25 over the supplied chunks. Call after ingestion changes."""
        self.hybrid.bm25_index.rebuild(chunks)
