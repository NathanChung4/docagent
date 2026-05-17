"""AST-based chunker for Python source.

Splits each Python file at top-level function and class boundaries so a chunk
is always a complete callable rather than a mid-function slice. Module-level
code (imports, constants, top-level statements) becomes its own chunk so
embedded constants and configuration values are still retrievable.

If parsing fails (malformed Python), falls back to one whole-file chunk so the
content is still indexable.
"""

from __future__ import annotations

import ast

from knowledge_rag.chunking.base import Chunker
from knowledge_rag.models import Chunk, Document, SourceType


class CodeChunker(Chunker):
    """Splits Python source by top-level function/class definitions."""

    def chunk(self, doc: Document) -> list[Chunk]:
        if doc.source_type != SourceType.CODE:
            return [self._make_chunk(doc, doc.content, "module", "<file>", 0)]

        source = doc.content
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [self._make_chunk(doc, source, "module", "<unparseable>", 0)]

        lines = source.splitlines(keepends=True)
        chunks: list[Chunk] = []

        # Identify top-level function/class spans. Anything outside these spans
        # is module-level code, collected separately so nothing is lost.
        spans: list[tuple[int, int, str, str]] = []  # (start, end, kind, name)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                continue
            start = node.lineno - 1
            end = (getattr(node, "end_lineno", None) or node.lineno) - 1
            spans.append((start, end, kind, node.name))

        # Emit module-level segments interleaved with function/class chunks in
        # source order so reading the chunks back gives you something close to
        # the original file's structure.
        cursor = 0
        chunk_index = 0
        spans.sort(key=lambda s: s[0])
        for start, end, kind, name in spans:
            if start > cursor:
                module_text = "".join(lines[cursor:start]).strip("\n")
                if module_text.strip():
                    chunks.append(
                        self._make_chunk(doc, module_text, "module", "<module>", chunk_index)
                    )
                    chunk_index += 1
            block_text = "".join(lines[start : end + 1]).rstrip("\n")
            chunks.append(self._make_chunk(doc, block_text, kind, name, chunk_index))
            chunk_index += 1
            cursor = end + 1

        # Trailing module-level code after the last def.
        if cursor < len(lines):
            tail = "".join(lines[cursor:]).strip("\n")
            if tail.strip():
                chunks.append(self._make_chunk(doc, tail, "module", "<module>", chunk_index))
                chunk_index += 1

        if not chunks:
            chunks.append(self._make_chunk(doc, source, "module", "<file>", 0))
        return chunks

    def _make_chunk(
        self,
        doc: Document,
        content: str,
        kind: str,
        symbol: str,
        index: int,
    ) -> Chunk:
        return Chunk.from_document(
            doc,
            content,
            {
                "strategy": "code",
                "kind": kind,
                "symbol": symbol,
                "chunk_index": index,
                **{k: v for k, v in doc.metadata.items() if k in {"filename", "module_docstring"}},
            },
        )
