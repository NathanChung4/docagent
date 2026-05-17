"""Sweep-report CSV loader.

Each CSV file becomes one Document. The rows themselves live in
`metadata['rows']`; StructuredChunker owns the text rendering used for
embedding. Document.content holds a minimal title line so empty-report
fallbacks have something to display.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from knowledge_rag.loaders.base import DirectoryLoader
from knowledge_rag.models import Document, SourceType


class SweepReportLoader(DirectoryLoader):
    """Loads parameter-sweep CSV reports."""

    glob_pattern = "*.csv"

    def _load_one(self, path: Path) -> Document:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: list[dict[str, Any]] = list(reader)
            columns = reader.fieldnames or []

        component = rows[0].get("component", "") if rows else ""
        title = f"{component} sweep report" if component else path.stem

        return Document(
            source_type=SourceType.REPORT,
            title=title,
            content=f"Sweep report: {title} ({len(rows)} rows)",
            uri=str(path.resolve()),
            metadata={
                "filename": path.name,
                "columns": columns,
                "row_count": len(rows),
                "rows": rows,
                "component": component,
            },
        )
