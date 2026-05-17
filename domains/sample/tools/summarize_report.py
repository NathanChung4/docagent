"""Summarize a sweep report CSV for a named component.

Picks the winning run by the component's primary metric (configured below)
and returns the row plus a few aggregate stats. Designed to be the structured
counterpart to a wiki-style answer: instead of telling the user "look at the
sweep report", the tool reads it and reports the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_rag.tools._helpers import (
    compute_param_ranges,
    find_latest_report,
    pick_winning_run,
    read_csv_rows,
)
from knowledge_rag.tools.base import Tool, ToolValidationError

# Per-component primary metric: (column_name, "max" or "min").
PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "flow_controller": ("throughput", "max"),
    "clock_divider": ("jitter_ps", "min"),
    "signal_buffer": ("dropped_units", "min"),
    "temp_regulator": ("overshoot", "min"),
    "pressure_sensor": ("event_latency_ms", "min"),
    "voltage_monitor": ("voltage_error", "min"),
}


class SummarizeReport(Tool):
    name = "summarize_report"
    description = (
        "Find the latest sweep report for a named component, identify the "
        "winning run by the component's primary metric, and return that row "
        "plus aggregate stats (run count, parameter ranges)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "Name of the component whose sweep report to summarize.",
            },
            "metric": {
                "type": "string",
                "description": (
                    "Optional metric to optimize. Defaults to the component's "
                    "primary metric (e.g., throughput for flow_controller)."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["max", "min"],
                "description": "Whether higher or lower is better. Defaults to the metric's natural direction.",
            },
        },
        "required": ["item_name"],
    }

    def __init__(self, sweep_reports_dir: Path) -> None:
        self.sweep_reports_dir = Path(sweep_reports_dir)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        item_name = kwargs.get("item_name")
        if not isinstance(item_name, str) or not item_name:
            raise ToolValidationError("'item_name' is required and must be a non-empty string.")

        report_path = find_latest_report(self.sweep_reports_dir, item_name)
        rows = read_csv_rows(report_path)
        if not rows:
            raise ToolValidationError(f"Sweep report {report_path.name} contains no data rows.")

        default_metric, default_direction = PRIMARY_METRIC.get(item_name, ("", "max"))
        metric = kwargs.get("metric") or default_metric
        direction = kwargs.get("direction") or default_direction
        if not metric:
            raise ToolValidationError(
                f"No primary metric configured for '{item_name}'. Pass 'metric' explicitly."
            )
        if metric not in rows[0]:
            raise ToolValidationError(
                f"Metric '{metric}' not present in {report_path.name}. "
                f"Available columns: {sorted(rows[0])}."
            )

        winner = pick_winning_run(rows, metric, direction)
        param_ranges = compute_param_ranges(rows, exclude={metric, "run_id", "component"})

        return {
            "status": "ok",
            "item_name": item_name,
            "report_file": report_path.name,
            "run_count": len(rows),
            "metric": metric,
            "direction": direction,
            "winning_run": winner,
            "param_ranges": param_ranges,
        }
