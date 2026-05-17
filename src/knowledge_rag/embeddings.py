"""Embedding generation using sentence-transformers.

Wraps a SentenceTransformer model behind a small, dependency-light interface so
the rest of the pipeline (vectorstore, retrieval) can stay model-agnostic.

Default model: `all-MiniLM-L6-v2` — 384 dimensions, ~80MB, fast on CPU. The
model is downloaded on first use and cached under the user's HF cache.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from knowledge_rag.models import Chunk

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class Embedder:
    """Embeds text into dense vectors via a sentence-transformers model.

    The underlying model is lazy-loaded on first encode call so importing this
    module is cheap and tests that don't touch embeddings stay fast.

    Attributes:
        model_name: Hugging Face id of the SentenceTransformer model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None  # SentenceTransformer once loaded

    def _load(self) -> Any:
        """Lazy-load the SentenceTransformer (downloads weights on first call)."""
        if self._model is None:
            # Imported lazily so module import doesn't pay the torch import cost.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g., 384 for all-MiniLM-L6-v2)."""
        model = self._load()
        # Newer sentence-transformers renamed this; keep both for forward/back compat.
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of strings. Returns a list of float lists (one per text)."""
        if not texts:
            return []
        model = self._load()
        # convert_to_numpy=True so we can call .tolist() and stay JSON-serializable
        # downstream. normalize_embeddings=True so cosine == dot product, which
        # ChromaDB's default L2 still ranks consistently with.
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_chunks(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        """Embed each chunk's content in place. Returns the same chunks list."""
        chunks_list = list(chunks)
        if not chunks_list:
            return chunks_list
        vectors = self.embed_texts([c.content for c in chunks_list])
        for chunk, vec in zip(chunks_list, vectors, strict=True):
            chunk.embedding = vec
        return chunks_list
