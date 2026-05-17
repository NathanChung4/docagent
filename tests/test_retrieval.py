"""Tests for the hybrid retrieval pipeline.

Mechanics-only tests on small synthetic corpora. Whether retrieval is *good*
on real documents is the eval suite's job (Phase 4.5), not these tests.
"""

from __future__ import annotations

from knowledge_rag.models import Chunk, RetrievalResult, SourceType
from knowledge_rag.retrieval import (
    BM25Index,
    HybridConfig,
    HybridRetriever,
    Retriever,
)
from knowledge_rag.vectorstore import VectorStore

# `store` and `shared_embedder` fixtures live in `tests/conftest.py`.


def _chunk(content: str, doc_id: str = "doc-1", **meta) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        content=content,
        source_type=meta.pop("source_type", SourceType.WIKI),
        title=meta.pop("title", "Sample"),
        uri=meta.pop("uri", "sample://x"),
        metadata=meta,
    )


# --- BM25Index ---------------------------------------------------------------


def test_bm25_empty_returns_no_hits() -> None:
    idx = BM25Index()
    assert idx.query("anything") == []
    assert len(idx) == 0


def test_bm25_rebuild_then_query_finds_keyword() -> None:
    idx = BM25Index()
    chunks = [
        _chunk("the threshold flag controls retry attempts"),
        _chunk("cookies should be baked at 350 degrees"),
        _chunk("widget power modes are documented in the user guide"),
    ]
    idx.rebuild(chunks)
    assert len(idx) == 3
    hits = idx.query("threshold flag", k=3)
    assert hits, "expected at least one BM25 hit"
    assert "threshold" in hits[0].chunk.content


def test_bm25_filter_drops_non_matching_chunks() -> None:
    idx = BM25Index()
    chunks = [
        _chunk("function definition for parser", source_type=SourceType.CODE),
        _chunk("function definition in the user guide", source_type=SourceType.WIKI),
    ]
    idx.rebuild(chunks)
    hits = idx.query("function definition", k=5, where={"source_type": "code"})
    assert len(hits) == 1
    assert hits[0].chunk.source_type == SourceType.CODE


def test_bm25_rebuild_replaces_corpus() -> None:
    idx = BM25Index()
    idx.rebuild([_chunk("first generation content")])
    assert len(idx) == 1
    idx.rebuild([_chunk("a"), _chunk("b"), _chunk("c")])
    assert len(idx) == 3


# --- HybridRetriever ---------------------------------------------------------


def _seed(store: VectorStore, chunks: list[Chunk]) -> BM25Index:
    """Add chunks to the vector store and return a BM25 index over the same set."""
    store.add_chunks(chunks)
    bm25 = BM25Index()
    bm25.rebuild(chunks)
    return bm25


def test_hybrid_returns_retrieval_results(store: VectorStore) -> None:
    chunks = [
        _chunk("memory controller frequency configuration"),
        _chunk("user guide for the widget power modes"),
    ]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    hits = hr.retrieve("memory controller", k=2)
    assert hits
    assert all(isinstance(h, RetrievalResult) for h in hits)


def test_hybrid_alpha_one_is_semantic_only(store: VectorStore) -> None:
    """alpha=1.0 → BM25 contributes nothing; ordering matches semantic-only."""
    chunks = [
        _chunk("how to configure the memory controller"),
        _chunk("an unrelated note about cookies"),
        _chunk("widget power modes overview"),
    ]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    hits = hr.retrieve("memory controller config", k=3, alpha=1.0)
    semantic_only = store.query("memory controller config", k=3)
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in semantic_only]


def test_hybrid_alpha_zero_is_bm25_only(store: VectorStore) -> None:
    """alpha=0.0 → semantic contributes nothing; ordering matches BM25-only."""
    chunks = [
        _chunk("the threshold flag is a tunable parameter"),
        _chunk("cookies are delicious and tasty"),
        _chunk("retry logic uses an exponential backoff schedule"),
    ]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    hits = hr.retrieve("threshold flag", k=3, alpha=0.0)
    bm25_only = bm25.query("threshold flag", k=3)
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in bm25_only]


def test_hybrid_blend_promotes_chunk_in_both_lists(store: VectorStore) -> None:
    """A chunk that scores well in *both* legs should beat single-leg winners.

    Constructing this carefully: 'threshold flag' is an exact-token match for
    chunk A (BM25 wins). 'memory controller config' is semantically close to
    chunk B. Chunk C contains both — should win the blend at alpha=0.5.
    """
    chunks = [
        _chunk("threshold flag tuning guide", doc_id="A"),
        _chunk("how to configure memory controller frequency", doc_id="B"),
        _chunk("threshold flag values when configuring the memory controller", doc_id="C"),
    ]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    hits = hr.retrieve("threshold flag memory controller", k=3, alpha=0.5)
    assert hits[0].chunk.doc_id == "C"


def test_hybrid_metadata_filter_applied_to_both_legs(store: VectorStore) -> None:
    chunks = [
        _chunk("widget configuration in code", source_type=SourceType.CODE),
        _chunk("widget configuration in the wiki", source_type=SourceType.WIKI),
    ]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    hits = hr.retrieve("widget configuration", k=5, where={"source_type": "code"})
    assert len(hits) == 1
    assert hits[0].chunk.source_type == SourceType.CODE


def test_hybrid_alpha_clamped(store: VectorStore) -> None:
    """alpha values outside [0,1] are clamped, not raised."""
    chunks = [_chunk("any content")]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    # Should not raise; treated as alpha=1.0.
    hits = hr.retrieve("content", k=1, alpha=5.0)
    assert hits
    hits = hr.retrieve("content", k=1, alpha=-1.0)
    assert hits


def test_hybrid_empty_query_returns_empty(store: VectorStore) -> None:
    chunks = [_chunk("anything")]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25)
    assert hr.retrieve("", k=5) == []


def test_hybrid_empty_corpus_returns_empty(store: VectorStore) -> None:
    bm25 = BM25Index()
    hr = HybridRetriever(store, bm25)
    assert hr.retrieve("anything", k=5) == []


def test_hybrid_config_alpha_default_used(store: VectorStore) -> None:
    chunks = [_chunk("threshold flag content"), _chunk("cookies content")]
    bm25 = _seed(store, chunks)
    hr = HybridRetriever(store, bm25, HybridConfig(alpha=0.0))
    # No alpha override → falls back to config (BM25-only).
    hits = hr.retrieve("threshold", k=2)
    assert "threshold" in hits[0].chunk.content


# --- Retriever facade --------------------------------------------------------


class _FakeReranker:
    """In-test stand-in for CrossEncoderReranker — promotes a target chunk_id.

    Lets us verify the facade actually invokes the reranker without paying the
    cost of loading a real cross-encoder model.
    """

    def __init__(self, promote_chunk_id: str) -> None:
        self.promote_chunk_id = promote_chunk_id
        self.last_call_n: int | None = None

    def rerank(self, query: str, results, top_n: int = 5):
        self.last_call_n = len(results)
        for r in results:
            r.rerank_score = 1.0 if r.chunk.chunk_id == self.promote_chunk_id else 0.0
        ordered = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
        return list(ordered[:top_n])


def test_retriever_without_reranker_returns_hybrid_topk(store: VectorStore) -> None:
    chunks = [
        _chunk("memory controller config", doc_id="A"),
        _chunk("widget power modes", doc_id="B"),
    ]
    bm25 = _seed(store, chunks)
    r = Retriever(store, bm25, reranker=None)
    hits = r.retrieve("memory controller", k=2)
    assert len(hits) == 2
    assert all(h.rerank_score is None for h in hits)


def test_retriever_with_reranker_uses_it(store: VectorStore) -> None:
    """Reranker should run on the over-fetched candidate pool."""
    target_id = "winner"
    chunks = [
        _chunk("widget power modes documentation", doc_id="A"),
        _chunk("memory controller frequency settings", doc_id="B"),
        Chunk(
            chunk_id=target_id,
            doc_id="C",
            content="an off-topic chunk that the fake reranker promotes anyway",
            source_type=SourceType.WIKI,
            title="t",
            uri="u",
        ),
    ]
    bm25 = _seed(store, chunks)
    fake = _FakeReranker(promote_chunk_id=target_id)
    r = Retriever(store, bm25, reranker=fake)
    hits = r.retrieve("memory controller", k=2)
    assert hits[0].chunk.chunk_id == target_id
    assert hits[0].rerank_score == 1.0
    # Facade should have over-fetched candidate_k for the reranker (capped by corpus size).
    assert fake.last_call_n is not None and fake.last_call_n >= len(chunks)


def test_retriever_use_rerank_false_skips_reranker(store: VectorStore) -> None:
    chunks = [_chunk("memory controller config", doc_id="A")]
    bm25 = _seed(store, chunks)
    fake = _FakeReranker(promote_chunk_id="never")
    r = Retriever(store, bm25, reranker=fake)
    r.retrieve("memory controller", k=1, use_rerank=False)
    assert fake.last_call_n is None  # never invoked


def test_retriever_rebuild_bm25_refits_index(store: VectorStore) -> None:
    bm25 = BM25Index()
    r = Retriever(store, bm25)
    chunks = [_chunk("first content"), _chunk("second content")]
    store.add_chunks(chunks)
    r.rebuild_bm25(chunks)
    assert len(bm25) == 2
