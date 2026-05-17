"""Chunker protocol and a source-type dispatcher.

Each concrete chunker handles one shape of content (prose, code, structured).
The dispatcher picks the right chunker based on Document.source_type so callers
can chunk a heterogeneous list of documents in one call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from knowledge_rag.models import Chunk, Document, SourceType


class Chunker(ABC):
    """Splits one Document into one or more Chunks.

    Implementations should:
      - Preserve source_type, title, and uri on every emitted Chunk.
      - Add per-chunk metadata that helps cite or filter the chunk later
        (e.g., section header, function name, row range).
      - Never silently drop content — if a chunker can't handle a document
        it should fall back to a single whole-document chunk rather than [].
    """

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split `doc` into chunks. Returns at least one chunk for non-empty docs."""


def chunk_document(doc: Document) -> list[Chunk]:
    """Dispatch to the right Chunker based on source_type.

    Default mapping:
      WIKI      -> ProseChunker
      CODE      -> CodeChunker
      REPORT    -> StructuredChunker (row batches)
      CHECKLIST -> StructuredChunker (single chunk; loader already split per row)
    """
    # Local imports avoid an import cycle with __init__.py at package import time.
    from knowledge_rag.chunking.code import CodeChunker
    from knowledge_rag.chunking.prose import ProseChunker
    from knowledge_rag.chunking.structured import StructuredChunker

    if doc.source_type == SourceType.WIKI:
        return ProseChunker().chunk(doc)
    if doc.source_type == SourceType.CODE:
        return CodeChunker().chunk(doc)
    if doc.source_type in (SourceType.REPORT, SourceType.CHECKLIST):
        return StructuredChunker().chunk(doc)
    # Unknown source type — fall back to a single chunk so nothing is lost.
    return [_whole_doc_chunk(doc)]


def chunk_documents(docs: Iterable[Document]) -> list[Chunk]:
    """Chunk a batch of Documents using the dispatcher. Order is preserved."""
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out


def _whole_doc_chunk(doc: Document) -> Chunk:
    """Fallback: emit the whole document as one chunk."""
    return Chunk.from_document(doc, doc.content, {"strategy": "whole_document"})
