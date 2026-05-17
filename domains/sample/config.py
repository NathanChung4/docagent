"""Sample-domain config: where the sample documents live and which tools to register."""

from __future__ import annotations

from pathlib import Path

from knowledge_rag.domain import DataSourcePaths, Domain
from knowledge_rag.tools.base import Tool

# Repo root is three levels up from this file: domains/sample/config.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_DATA = _REPO_ROOT / "data" / "sample"
_GENERATED_DIR = _SAMPLE_DATA / "generated"


class SampleDomain(Domain):
    """Public-portfolio domain pack. Generic widget/component vocabulary."""

    name = "sample"

    def paths(self) -> DataSourcePaths:
        return DataSourcePaths(
            confluence_dir=_SAMPLE_DATA / "confluence",
            github_dir=_SAMPLE_DATA / "github",
            sweep_reports_dir=_SAMPLE_DATA / "sweep_reports",
            checklist_path=_SAMPLE_DATA / "checklist" / "components.xlsx",
        )

    def tools(self) -> list[Tool]:
        # Imported lazily so the core can introspect Domain without paying the
        # tool import cost when tools aren't needed (e.g., during ingestion).
        from domains.sample.tools.generate_config_file import GenerateConfigFile
        from domains.sample.tools.lookup_item_status import LookupItemStatus
        from domains.sample.tools.summarize_report import SummarizeReport

        paths = self.paths()
        return [
            GenerateConfigFile(output_dir=_GENERATED_DIR),
            SummarizeReport(sweep_reports_dir=paths.sweep_reports_dir),
            LookupItemStatus(checklist_path=paths.checklist_path),
        ]
