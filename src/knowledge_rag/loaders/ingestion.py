"""Unified ingestion entry point.

`ingest_all(domain)` reads the domain's DataSourcePaths and runs every loader
against the matching directory/file, returning the combined list of Documents.
"""

from __future__ import annotations

from knowledge_rag.domain import Domain
from knowledge_rag.loaders.checklist import ChecklistLoader
from knowledge_rag.loaders.confluence import ConfluenceLoader
from knowledge_rag.loaders.github import GitHubLoader
from knowledge_rag.loaders.sweep_report import SweepReportLoader
from knowledge_rag.models import Document


def ingest_all(domain: Domain) -> list[Document]:
    """Run every loader against the active domain's source paths.

    Args:
        domain: The active Domain instance (usually from get_domain()).

    Returns:
        Flat list of Documents from all four source types, in stable order:
        wiki -> code -> reports -> checklist.
    """
    paths = domain.paths()
    docs: list[Document] = []
    docs.extend(ConfluenceLoader().load(paths.confluence_dir))
    docs.extend(GitHubLoader().load(paths.github_dir))
    docs.extend(SweepReportLoader().load(paths.sweep_reports_dir))
    docs.extend(ChecklistLoader().load(paths.checklist_path))
    return docs
