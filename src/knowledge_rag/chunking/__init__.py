"""Chunking strategies — turn Documents into retrievable Chunks.

A chunker is responsible for one source format. Pick the right chunker for a
given document by source_type via `chunk_document()`, or call a chunker directly
when you know the type.
"""

from __future__ import annotations

from knowledge_rag.chunking.base import Chunker, chunk_document, chunk_documents
from knowledge_rag.chunking.code import CodeChunker
from knowledge_rag.chunking.prose import ProseChunker
from knowledge_rag.chunking.structured import StructuredChunker

__all__ = [
    "Chunker",
    "ProseChunker",
    "CodeChunker",
    "StructuredChunker",
    "chunk_document",
    "chunk_documents",
]
