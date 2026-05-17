"""Look up owner / status / due date / notes for a checklist item.

Distinct from the wiki-Q&A path: this is a structured lookup against a single
source of truth rather than free-form retrieval over chunked text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_rag.tools._helpers import lookup_xlsx_row
from knowledge_rag.tools.base import Tool, ToolValidationError


class LookupItemStatus(Tool):
    name = "lookup_item_status"
    description = (
        "Look up the owner, status, due date, priority, and notes for a "
        "named item from the project checklist. Returns 'found: false' if "
        "the item isn't in the checklist."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "Name of the checklist item (e.g., 'flow_controller').",
            },
        },
        "required": ["item_name"],
    }

    def __init__(self, checklist_path: Path) -> None:
        self.checklist_path = Path(checklist_path)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        item_name = kwargs.get("item_name")
        if not isinstance(item_name, str) or not item_name:
            raise ToolValidationError("'item_name' is required and must be a non-empty string.")

        fields = lookup_xlsx_row(self.checklist_path, item_name)
        if fields is None:
            return {"status": "ok", "found": False, "item_name": item_name}
        return {
            "status": "ok",
            "found": True,
            "item_name": item_name,
            "fields": fields,
        }
