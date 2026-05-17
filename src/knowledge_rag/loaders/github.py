"""GitHub / local Python script loader.

Captures the full source plus the module docstring and a list of top-level
function/class names. The AST-based code chunker in Phase 2 uses the same
structural information to split files at function boundaries.
"""

from __future__ import annotations

import ast
from pathlib import Path

from knowledge_rag.loaders.base import DirectoryLoader
from knowledge_rag.models import Document, SourceType


class GitHubLoader(DirectoryLoader):
    """Loads .py files into Documents preserving their full source.

    Metadata captured per file:
      - module_docstring: top-of-file docstring if present
      - functions: list of top-level function names
      - classes: list of top-level class names
      - parse_error: present iff the file failed to parse; AST metadata empty
    """

    glob_pattern = "*.py"

    def _load_one(self, path: Path) -> Document:
        source = path.read_text(encoding="utf-8")

        module_docstring = ""
        functions: list[str] = []
        classes: list[str] = []
        parse_error: str | None = None
        try:
            tree = ast.parse(source)
            module_docstring = ast.get_docstring(tree) or ""
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except SyntaxError as exc:
            # Surface the failure in metadata rather than silently swallowing
            # it. The code chunker has its own SyntaxError fallback to a single
            # whole-file chunk, so ingestion still succeeds.
            parse_error = f"{exc.__class__.__name__}: {exc}"

        metadata: dict = {
            "filename": path.name,
            "module_docstring": module_docstring,
            "functions": functions,
            "classes": classes,
        }
        if parse_error is not None:
            metadata["parse_error"] = parse_error

        return Document(
            source_type=SourceType.CODE,
            title=path.stem,
            content=source,
            uri=str(path.resolve()),
            metadata=metadata,
        )
