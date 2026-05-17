"""Tests for the sentence-transformers Embedder wrapper.

These tests exercise the real model (not mocks) because the contract we care
about is "does it produce vectors that mean what we think they mean?". The
model downloads on first run; subsequent runs use the HF cache.
"""

from __future__ import annotations

import math

import pytest

from knowledge_rag.embeddings import DEFAULT_DIM, Embedder
from knowledge_rag.models import Chunk, SourceType


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Embedder normalizes, so this also equals dot product."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Module-scoped so the model loads once across all tests in this file."""
    return Embedder()


def test_embed_texts_returns_correct_shape(embedder: Embedder) -> None:
    vectors = embedder.embed_texts(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert all(len(v) == DEFAULT_DIM for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_embed_empty_input_returns_empty_list(embedder: Embedder) -> None:
    assert embedder.embed_texts([]) == []


def test_similar_sentences_score_higher_than_dissimilar(embedder: Embedder) -> None:
    """The whole point of embeddings: semantic similarity shows up as cosine."""
    a, b, c = embedder.embed_texts(
        [
            "How do I configure the memory controller settings?",
            "What are the configuration options for the memory controller?",
            "The recipe calls for two cups of flour and a pinch of salt.",
        ]
    )
    sim_ab = _cosine(a, b)  # paraphrases — should be close
    sim_ac = _cosine(a, c)  # unrelated topics — should be far
    assert sim_ab > sim_ac + 0.2, f"expected gap, got sim_ab={sim_ab}, sim_ac={sim_ac}"


def test_dim_matches_model(embedder: Embedder) -> None:
    assert embedder.dim == DEFAULT_DIM


def test_embed_chunks_sets_embedding_in_place(embedder: Embedder) -> None:
    chunks = [
        Chunk(
            doc_id="d1",
            content=f"chunk number {i}",
            source_type=SourceType.WIKI,
            title="t",
            uri="u",
        )
        for i in range(3)
    ]
    out = embedder.embed_chunks(chunks)
    assert out is not chunks  # returns a list, not the original iterable
    assert all(c.embedding is not None for c in chunks)
    assert all(len(c.embedding) == DEFAULT_DIM for c in chunks)  # type: ignore[arg-type]


def test_embed_chunks_handles_empty(embedder: Embedder) -> None:
    assert embedder.embed_chunks([]) == []
