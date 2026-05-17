"""Shared eval-pipeline plumbing used by both `scripts/run_eval.py` and `scripts/run_sweep.py`.

These helpers were originally inlined in `run_eval.py` (Phase 4.5/4.7). When
Phase 9 added the parameter sweep runner, both scripts needed the same
ingest-or-reuse logic and the same retriever-construction wrappers — pulling
them up to a shared module keeps the scripts thin and the construction logic
single-sourced.

What lives here:
  - `ensure_index`: ingest + chunk + (re)build the pgvector store and BM25 index
    for the active domain. Idempotent: if the table already has chunks, it skips
    re-embedding (the slow part).
  - `build_retriever`: assemble a `Retriever` with optional reranker from a set
    of hyperparameters. The sweep runner calls this many times with different
    cells; the eval runner calls it once.
  - `TimedRetriever`: thin facade that records per-`retrieve()` wall-clock so
    callers can compute p50/p95 without bleeding timing into the eval module.
  - `latency_summary`: rounded p50 / p95 / mean over a list of latencies.

Stays generic — zero domain-specific logic. The active domain is supplied as a
name (or via `KNOWLEDGE_DOMAIN`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from knowledge_rag.chunking import chunk_documents
from knowledge_rag.domain import get_domain
from knowledge_rag.loaders.ingestion import ingest_all
from knowledge_rag.models import RetrievalResult
from knowledge_rag.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from knowledge_rag.retrieval import (
    DEFAULT_ALPHA,
    DEFAULT_CANDIDATE_K,
    BM25Index,
    HybridConfig,
    Retriever,
)
from knowledge_rag.vectorstore import VectorStore

# DEFAULT_RERANKER_MODEL is re-exported above (imported from reranker) so existing
# `from knowledge_rag.eval_pipeline import DEFAULT_RERANKER_MODEL` callsites in
# scripts/ keep working — single source of truth lives in reranker.py.


@dataclass
class CorpusStats:
    """Snapshot of what was indexed — written into result JSON for reproducibility."""

    n_documents: int
    n_chunks: int


def ensure_index(
    domain_name: str,
    dsn: str,
    table: str,
    *,
    reset: bool = False,
) -> tuple[VectorStore, BM25Index, CorpusStats]:
    """Build (or reuse) the vector store + BM25 index for `domain_name`.

    `reset=True` drops the table first — use when you want to force a re-embed
    (e.g., the embedding model changed). Otherwise the existing table is reused
    if it already has rows, which is the common case during eval iteration.

    Returns the store, BM25 index, and corpus stats. The corpus stats are
    captured here (not in the caller) because re-deriving them later would
    require re-running the loaders.
    """
    store = VectorStore(dsn=dsn, table_name=table)
    bm25 = BM25Index()

    if reset:
        store.reset()

    domain = get_domain(domain_name)
    docs = ingest_all(domain)
    chunks = chunk_documents(docs)

    if store.count() == 0 and chunks:
        store.add_chunks(chunks)
    bm25.rebuild(chunks)

    return store, bm25, CorpusStats(n_documents=len(docs), n_chunks=len(chunks))


def build_retriever(
    store: VectorStore,
    bm25: BM25Index,
    *,
    alpha: float = DEFAULT_ALPHA,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    use_rerank: bool = True,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker: CrossEncoderReranker | None = None,
) -> Retriever:
    """Assemble a Retriever from explicit hyperparameters.

    Pass `reranker` to reuse an already-loaded CrossEncoderReranker (the sweep
    runner does this — the model is ~200MB and reloading per cell adds real
    wall time). When omitted and `use_rerank=True`, a fresh reranker is
    constructed.
    """
    if use_rerank:
        active_reranker = (
            reranker if reranker is not None else CrossEncoderReranker(model_name=reranker_model)
        )
    else:
        active_reranker = None
    return Retriever(
        vector_store=store,
        bm25_index=bm25,
        reranker=active_reranker,
        hybrid_config=HybridConfig(candidate_k=candidate_k, alpha=alpha),
        candidate_k=candidate_k,
    )


class TimedRetriever:
    """Retriever facade that records wall-clock latency per `retrieve()` call.

    Wrapped around the real Retriever before handing it to `evaluate_retriever`
    so we can compute p50/p95 query latency without bleeding timing logic into
    the generic eval module.
    """

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.latencies_ms: list[float] = []

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        t0 = time.perf_counter()
        out = self.inner.retrieve(query, k=k, **kwargs)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return out


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """p50 / p95 / mean over the per-query latencies. Empty list → all zeros."""
    if not latencies_ms:
        return {"n_queries": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    p50, p95 = np.percentile(latencies_ms, [50, 95])
    return {
        "n_queries": len(latencies_ms),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
    }
