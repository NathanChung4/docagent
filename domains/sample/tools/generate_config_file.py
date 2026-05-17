"""Generate a CSV config file for a named component.

Validation strategy: each component has a hardcoded spec in KNOWN_SPECS,
mirroring what the wiki page documents. A future iteration can swap this for
a parser that extracts the ranges from the retrieved spec text — the call
surface stays the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_rag.tools._helpers import (
    EnumSpec,
    ParamSpec,
    SpecTable,
    validate_params,
    write_kv_csv,
)
from knowledge_rag.tools.base import Tool, ToolValidationError

# Mirror data/sample/confluence/*.html. Update both sides if either changes.
KNOWN_SPECS: dict[str, SpecTable] = {
    "flow_controller": {
        "rate_limit": ParamSpec(int, 1, 1000, 100),
        "burst_size": ParamSpec(int, 1, 500, 50),
        "backoff_ms": ParamSpec(int, 0, 5000, 250),
    },
    "clock_divider": {
        "divisor": ParamSpec(int, 1, 256, 8),
        "jitter_budget_ps": ParamSpec(int, 10, 10_000, 500),
        "phase_offset_deg": ParamSpec(float, 0.0, 359.9, 0.0),
    },
    "pressure_sensor": {
        "sample_interval_ms": ParamSpec(int, 10, 10_000, 100),
        "warn_threshold": ParamSpec(int, 0, 100, 70),
        "crit_threshold": ParamSpec(int, 0, 100, 90),
    },
    "signal_buffer": {
        "buffer_depth": ParamSpec(int, 1, 4096, 256),
        "drop_policy": EnumSpec(frozenset({"oldest", "newest", "block"}), default="newest"),
    },
    "temp_regulator": {
        "kp": ParamSpec(float, 0.0, 10.0, 1.0),
        "ki": ParamSpec(float, 0.0, 5.0, 0.1),
        "kd": ParamSpec(float, 0.0, 5.0, 0.05),
    },
    "voltage_monitor": {
        "sample_rate_hz": ParamSpec(int, 1, 10_000, 100),
        "voltage_min": ParamSpec(float, 0.0, 5.0, 0.9),
        "voltage_max": ParamSpec(float, 0.0, 5.0, 1.1),
    },
}


class GenerateConfigFile(Tool):
    name = "generate_config_file"
    description = (
        "Generate a CSV configuration file for a named component using the "
        "supplied parameter values. Validates parameter names, types, and "
        "ranges against the component's specification before writing. Returns "
        "the absolute path of the file written."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": (
                    "Name of the component (e.g., 'flow_controller', "
                    "'clock_divider', 'pressure_sensor')."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Parameter name -> value mapping. Only parameters listed "
                    "in the component spec are accepted; values must fall "
                    "inside the spec's allowed range."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["item_name", "params"],
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        item_name = kwargs.get("item_name")
        params = kwargs.get("params")
        if not isinstance(item_name, str) or not item_name:
            raise ToolValidationError("'item_name' is required and must be a non-empty string.")
        if not isinstance(params, dict):
            raise ToolValidationError("'params' is required and must be an object.")

        spec = KNOWN_SPECS.get(item_name)
        if spec is None:
            raise ToolValidationError(
                f"Unknown component '{item_name}'. Known components: "
                f"{', '.join(sorted(KNOWN_SPECS))}."
            )
        validate_params(item_name, params, spec)

        out_path = write_kv_csv(self.output_dir, item_name, params)
        return {
            "status": "ok",
            "item_name": item_name,
            "params_written": params,
            "output_path": str(out_path),
        }
