"""Domain interface and the get_domain() factory.

A 'domain pack' bundles three things: data-source paths (config), the tools the
agent can call, and an evaluation dataset. The generic core consumes a Domain
through this interface and never imports a specific pack — the pack is selected
at runtime by the KNOWLEDGE_DOMAIN environment variable.

This is the seam that makes the public-portfolio release a publish, not a rewrite.
"""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowledge_rag.tools.base import Tool


@dataclass
class DataSourcePaths:
    """Where each loader should look for its source files.

    Domain packs populate this from their own config so the loaders themselves
    stay source-agnostic — a loader knows how to parse HTML, not which folder
    holds the HTML for any particular domain.
    """

    confluence_dir: Path
    github_dir: Path
    sweep_reports_dir: Path
    checklist_path: Path

    extras: dict[str, Path] = field(default_factory=dict)


class Domain(ABC):
    """Abstract base class for a domain pack.

    Each pack subclasses this and supplies:
      - name: short identifier matching the package name (e.g., "sample")
      - paths(): where to find source files
      - tools(): tool instances the agent can call
      - eval_dataset(): question/expected-answer pairs for Phase 9 eval
    """

    name: str = ""

    @abstractmethod
    def paths(self) -> DataSourcePaths:
        """Return the data-source paths used by the loaders."""

    @abstractmethod
    def tools(self) -> list[Tool]:
        """Return the tools available to the agent in this domain."""

    def eval_dataset(self) -> list[dict[str, Any]]:
        """Load and return the eval dataset (default: read eval_dataset.json next to config.py).

        Domain packs can override if their dataset lives elsewhere.
        """
        pkg = importlib.import_module(f"domains.{self.name}")
        pkg_dir = Path(pkg.__file__).parent  # type: ignore[arg-type]
        dataset_path = pkg_dir / "eval_dataset.json"
        if not dataset_path.exists():
            return []
        return json.loads(dataset_path.read_text(encoding="utf-8"))


class DomainNotFoundError(RuntimeError):
    """Raised when KNOWLEDGE_DOMAIN points to a pack that can't be loaded."""


def get_domain(name: str | None = None) -> Domain:
    """Resolve and instantiate the active domain pack.

    Resolution order: explicit `name` arg → KNOWLEDGE_DOMAIN env var → "sample".
    The chosen pack must be importable as `domains.{name}` and must export a
    callable `get_domain` that returns a Domain instance.

    Args:
        name: Override the env var. Used in tests to force a specific pack.

    Returns:
        An instance of the selected pack's Domain subclass.

    Raises:
        DomainNotFoundError: pack import fails or doesn't expose get_domain().
    """
    pack_name = name or os.environ.get("KNOWLEDGE_DOMAIN", "sample")
    try:
        pack = importlib.import_module(f"domains.{pack_name}")
    except ImportError as e:
        raise DomainNotFoundError(
            f"Could not import domain pack 'domains.{pack_name}': {e}. "
            "Set KNOWLEDGE_DOMAIN to a valid pack name."
        ) from e

    factory = getattr(pack, "get_domain", None)
    if factory is None:
        raise DomainNotFoundError(
            f"Domain pack 'domains.{pack_name}' must export a get_domain() factory."
        )
    return factory()
