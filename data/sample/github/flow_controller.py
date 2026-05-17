"""Flow controller validation script.

Drives a synthetic workload through the flow controller and reports throughput,
latency, and dropped-unit counts. Intended to be run by the sweep harness with
parameters injected via environment variables.
"""

from __future__ import annotations

import os

RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "100"))
BURST_SIZE = int(os.environ.get("BURST_SIZE", "50"))
BACKOFF_MS = int(os.environ.get("BACKOFF_MS", "250"))


def configure() -> dict[str, int]:
    """Apply the parameter set to the flow controller and return the active config."""
    return {
        "rate_limit": RATE_LIMIT,
        "burst_size": BURST_SIZE,
        "backoff_ms": BACKOFF_MS,
    }


def run_sweep(work_units: int = 10_000) -> dict[str, float]:
    """Push `work_units` through the controller and capture telemetry.

    Returns a dict of metric -> measured value. The sweep harness writes this
    to the report CSV for downstream analysis.
    """
    cfg = configure()
    # Real implementation would talk to the device under test; this stub returns
    # plausible numbers so the rest of the pipeline can be exercised.
    throughput = min(cfg["rate_limit"], work_units / 60.0)
    latency_p99_ms = 50.0 + max(0, cfg["burst_size"] - cfg["rate_limit"]) * 0.5
    dropped = max(0, work_units - int(throughput * 60))
    return {
        "throughput": throughput,
        "latency_p99_ms": latency_p99_ms,
        "dropped_units": float(dropped),
    }


if __name__ == "__main__":
    print(run_sweep())
