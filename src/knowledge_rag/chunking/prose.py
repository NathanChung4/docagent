"""Prose chunker for wiki-style documents.

Splits a wiki Document by section headers (captured into Document.metadata
during loading). Each emitted chunk carries the section title in its metadata
so retrieval can filter or cite by section.

If a single section's body exceeds `max_chars`, that section is further split
into overlapping word-level windows so no chunk grows unbounded.
"""

from __future__ import annotations

from knowledge_rag.chunking.base import Chunker
from knowledge_rag.models import Chunk, Document, SourceType


class ProseChunker(Chunker):
    """Section-aware splitter for HTML/wiki content.

    Args:
        max_chars: Soft upper bound on chunk size. Sections smaller than this
            stay intact; larger sections are window-split.
        overlap_chars: When a section is window-split, each subsequent window
            repeats roughly this many trailing characters of the previous
            window so a sentence cut in half is still searchable from both sides.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 150) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        if doc.source_type != SourceType.WIKI:
            # Defensive: still produce something useful rather than crashing.
            return [self._make_chunk(doc, doc.title or "document", doc.content, 0)]

        sections = doc.metadata.get("sections") or []
        body = doc.content or ""

        # Build (section_title, section_body) pairs by walking the text and
        # treating any line that exactly matches a known section title as a
        # boundary. Anything before the first boundary becomes a "preamble"
        # section keyed by the document title.
        section_set = {s for s in sections if s}
        chunks: list[Chunk] = []
        current_title = doc.title or "preamble"
        current_lines: list[str] = []
        chunk_index = 0

        def flush() -> None:
            nonlocal chunk_index, current_lines
            text = "\n".join(line for line in current_lines if line.strip())
            if not text.strip():
                current_lines = []
                return
            for piece in self._split_oversized(text):
                chunks.append(self._make_chunk(doc, current_title, piece, chunk_index))
                chunk_index += 1
            current_lines = []

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if line in section_set:
                flush()
                current_title = line
                continue
            current_lines.append(raw_line)
        flush()

        # If a doc had no detected sections at all, still emit one chunk.
        if not chunks and body.strip():
            chunks.append(self._make_chunk(doc, doc.title or "document", body, 0))
        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        """Window-split text larger than max_chars with character-level overlap.

        Uses character counts rather than tokens because we don't depend on a
        tokenizer here — Phase 3's embedder owns tokenization. char/4 is a
        decent stand-in for token count.
        """
        if len(text) <= self.max_chars:
            return [text]
        pieces: list[str] = []
        start = 0
        step = self.max_chars - self.overlap_chars
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            pieces.append(text[start:end])
            if end == len(text):
                break
            start += step
        return pieces

    def _make_chunk(self, doc: Document, section: str, content: str, index: int) -> Chunk:
        return Chunk.from_document(
            doc,
            content,
            {
                "strategy": "prose",
                "section": section,
                "chunk_index": index,
                **{k: v for k, v in doc.metadata.items() if k != "sections"},
            },
        )
