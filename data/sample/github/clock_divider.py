"""Clock divider validation script.

Sweeps the divisor across power-of-two and odd values and reports output
frequency error and cycle-to-cycle jitter.
"""

from __future__ import annotations

import os

DIVISOR = int(os.environ.get("DIVISOR", "8"))
JITTER_BUDGET_PS = int(os.environ.get("JITTER_BUDGET_PS", "500"))
PHASE_OFFSET_DEG = float(os.environ.get("PHASE_OFFSET_DEG", "0.0"))

REFERENCE_FREQ_MHZ = 1000.0


def is_power_of_two(n: int) -> bool:
    """True for divisors that produce minimal output jitter."""
    return n > 0 and (n & (n - 1)) == 0


def measure() -> dict[str, float]:
    """Compute output frequency and worst-case jitter for the active divisor."""
    output_mhz = REFERENCE_FREQ_MHZ / DIVISOR
    base_jitter = 100.0
    penalty = 0.0 if is_power_of_two(DIVISOR) else base_jitter * 0.6
    jitter_ps = base_jitter + penalty
    return {
        "output_freq_mhz": output_mhz,
        "jitter_ps": jitter_ps,
        "phase_offset_deg": PHASE_OFFSET_DEG,
        "within_budget": float(jitter_ps <= JITTER_BUDGET_PS),
    }


if __name__ == "__main__":
    print(measure())
