"""Chunker for structured row data (sweep CSVs, checklist rows).

Two distinct shapes share this chunker:

  REPORT (sweep CSV): the loader emits one Document per file with all rows in
    metadata['rows']. Splitting per-row is too granular for retrieval — a single
    row has no semantic context. Instead we batch rows together and prepend the
    column header so each chunk is a self-contained mini-table.

  CHECKLIST (xlsx): the loader already emits one Document per row, so each
    document becomes exactly one chunk. The structured fields are pulled into
    chunk metadata for filtering (e.g., "owner = jane").
"""

from __future__ import annotations

from typing import Any

from knowledge_rag.chunking.base import Chunker
from knowledge_rag.models import Chunk, Document, SourceType


class StructuredChunker(Chunker):
    """Row-batch chunker for tabular sources.

    Args:
        rows_per_chunk: How many sweep rows to bundle into one chunk. Larger
            values give the embedder more context per chunk; smaller values
            give finer retrieval granularity.
        row_overlap: How many rows of the previous batch to repeat at the start
            of the next batch. Helps catch cross-batch trends in retrieval.
    """

    def __init__(self, rows_per_chunk: int = 10, row_overlap: int = 1) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        if row_overlap < 0 or row_overlap >= rows_per_chunk:
            raise ValueError("row_overlap must be in [0, rows_per_chunk)")
        self.rows_per_chunk = rows_per_chunk
        self.row_overlap = row_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        if doc.source_type == SourceType.CHECKLIST:
            return self._chunk_checklist(doc)
        if doc.source_type == SourceType.REPORT:
            return self._chunk_report(doc)
        return [Chunk.from_document(doc, doc.content, {"strategy": "structured_passthrough"})]

    def _chunk_checklist(self, doc: Document) -> list[Chunk]:
        """One chunk per Document — the loader already split per row."""
        fields = doc.metadata.get("fields", {}) or {}
        return [
            Chunk.from_document(
                doc,
                doc.content,
                {
                    "strategy": "checklist_row",
                    "chunk_index": 0,
                    "row_index": doc.metadata.get("row_index"),
                    "item_id": doc.metadata.get("item_id"),
                    # Surface fields one level up so retrieval can filter on them
                    # (e.g., status="active") without parsing the content blob.
                    **{f"field_{k}": v for k, v in fields.items()},
                },
            )
        ]

    def _chunk_report(self, doc: Document) -> list[Chunk]:
        """Batch CSV rows into chunks, prepending the header to each batch."""
        rows: list[dict[str, Any]] = doc.metadata.get("rows") or []
        columns: list[str] = doc.metadata.get("columns") or []
        component = doc.metadata.get("component", "")

        if not rows:
            return [
                Chunk.from_document(
                    doc,
                    doc.content,
                    {"strategy": "report_empty", "chunk_index": 0},
                )
            ]

        chunks: list[Chunk] = []
        chunk_index = 0
        step = self.rows_per_chunk - self.row_overlap
        i = 0
        while i < len(rows):
            batch = rows[i : i + self.rows_per_chunk]
            text = self._render_batch(doc.title, columns, batch, i)
            chunks.append(
                Chunk.from_document(
                    doc,
                    text,
                    {
                        "strategy": "report_rows",
                        "chunk_index": chunk_index,
                        "row_start": i,
                        "row_end": i + len(batch) - 1,
                        "row_count": len(batch),
                        "columns": columns,
                        "component": component,
                    },
                )
            )
            chunk_index += 1
            if i + self.rows_per_chunk >= len(rows):
                break
            i += step
        return chunks

    @staticmethod
    def _render_batch(
        title: str, columns: list[str], rows: list[dict[str, Any]], offset: int
    ) -> str:
        header = f"Sweep report: {title}"
        cols = f"Columns: {', '.join(columns)}" if columns else ""
        body_lines = []
        for j, row in enumerate(rows):
            pairs = ", ".join(f"{k}={row.get(k, '')}" for k in (columns or row.keys()))
            body_lines.append(f"row {offset + j}: {pairs}")
        return "\n".join([header, cols, "", *body_lines]).strip()
