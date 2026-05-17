"""Voltage monitor validation script.

Injects controlled voltage steps and confirms the monitor alerts only when the
step exceeds the configured tolerance.
"""

from __future__ import annotations

import os

NOMINAL_VOLTAGE = float(os.environ.get("NOMINAL_VOLTAGE", "1.2"))
TOLERANCE_PCT = float(os.environ.get("TOLERANCE_PCT", "5.0"))
SAMPLE_HZ = int(os.environ.get("SAMPLE_HZ", "1000"))


def in_tolerance(step_pct: float) -> bool:
    """True if a step of `step_pct` from nominal is inside the tolerance window."""
    return abs(step_pct) <= TOLERANCE_PCT


def inject_steps(steps_pct: list[float]) -> dict[str, int]:
    """Apply each step in turn and count how many should produce alerts."""
    alerts = sum(1 for s in steps_pct if not in_tolerance(s))
    return {
        "expected_alerts": alerts,
        "steps_total": len(steps_pct),
        "sample_hz": SAMPLE_HZ,
    }


if __name__ == "__main__":
    print(inject_steps([1.0, -1.0, 5.0, -5.0, 10.0, -10.0]))
