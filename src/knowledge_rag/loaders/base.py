"""Loader protocol shared by all source-format adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from knowledge_rag.models import Document


class LoaderError(Exception):
    """Raised when a loader hits a malformed source it can't recover from.

    Missing-source paths are NOT a LoaderError — loaders return [] for absent
    inputs so partial domain configs still ingest. This exception is for the
    case where a file *exists* but its contents are unusable.
    """


class Loader(ABC):
    """A loader knows how to turn one source-format directory or file into Documents.

    Loaders are source-agnostic — they understand HTML or CSV or .py, not the
    domain those files describe. Domain packs supply the paths via DataSourcePaths.
    """

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """Load every document under `path` and return them in a list."""


class DirectoryLoader(Loader):
    """Loader for a directory of files matching a glob.

    Subclasses set `glob_pattern` (e.g. "*.html") and implement `_load_one(path)`.
    The base class handles the directory-scan boilerplate that was duplicated
    across three loaders pre-refactor.

    Missing or non-directory paths return [] (unconfigured source).
    """

    glob_pattern: str = ""

    def load(self, path: Path) -> list[Document]:
        if not path.exists() or not path.is_dir():
            return []
        if not self.glob_pattern:
            raise LoaderError(f"{type(self).__name__} did not set glob_pattern")
        return [self._load_one(p) for p in sorted(path.glob(self.glob_pattern))]

    @abstractmethod
    def _load_one(self, path: Path) -> Document:
        """Convert one matched file into a Document."""
