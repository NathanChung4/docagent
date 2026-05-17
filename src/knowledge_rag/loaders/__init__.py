"""Document loaders. One module per source format; ingestion.py wires them together."""

from knowledge_rag.loaders.base import Loader
from knowledge_rag.loaders.checklist import ChecklistLoader
from knowledge_rag.loaders.confluence import ConfluenceLoader
from knowledge_rag.loaders.github import GitHubLoader
from knowledge_rag.loaders.ingestion import ingest_all
from knowledge_rag.loaders.sweep_report import SweepReportLoader

__all__ = [
    "Loader",
    "ChecklistLoader",
    "ConfluenceLoader",
    "GitHubLoader",
    "SweepReportLoader",
    "ingest_all",
]
