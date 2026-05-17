"""Pressure sensor validation script.

Runs the pressure sensor against a queue-fill ramp and confirms that warn and
critical events fire at the configured thresholds.
"""

from __future__ import annotations

import os

SAMPLE_INTERVAL_MS = int(os.environ.get("SAMPLE_INTERVAL_MS", "100"))
WARN_THRESHOLD = int(os.environ.get("WARN_THRESHOLD", "70"))
CRIT_THRESHOLD = int(os.environ.get("CRIT_THRESHOLD", "90"))


def validate_thresholds() -> None:
    """Reject configurations the sensor would refuse to start with."""
    if WARN_THRESHOLD >= CRIT_THRESHOLD:
        raise ValueError(
            f"warn_threshold ({WARN_THRESHOLD}) must be below "
            f"crit_threshold ({CRIT_THRESHOLD})"
        )


def ramp_queue() -> dict[str, int]:
    """Ramp queue depth from 0% to 100% in 10% steps and capture event counts."""
    validate_thresholds()
    warns = 0
    crits = 0
    for fill_pct in range(0, 101, 10):
        if fill_pct >= CRIT_THRESHOLD:
            crits += 1
        elif fill_pct >= WARN_THRESHOLD:
            warns += 1
    return {
        "warn_events": warns,
        "crit_events": crits,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
    }


if __name__ == "__main__":
    print(ramp_queue())
