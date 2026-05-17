"""Confluence/wiki HTML loader. Strips markup and captures section structure."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from knowledge_rag.loaders.base import DirectoryLoader
from knowledge_rag.models import Document, SourceType

_HEADER_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


class ConfluenceLoader(DirectoryLoader):
    """Loads HTML wiki pages and converts each into a single plain-text Document.

    Section headers (h1–h6) are preserved inline so the prose chunker in Phase 2
    can split on them. A list of section titles is captured in metadata for
    later filtering and display.
    """

    glob_pattern = "*.html"

    def _load_one(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "html.parser")

        title_tag = soup.find("title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else path.stem

        sections = [h.get_text(strip=True) for h in soup.find_all(_HEADER_TAGS)]

        body = soup.body or soup
        text = body.get_text(separator="\n", strip=True)

        return Document(
            source_type=SourceType.WIKI,
            title=title,
            content=text,
            uri=str(path.resolve()),
            metadata={
                "filename": path.name,
                "sections": sections,
            },
        )
