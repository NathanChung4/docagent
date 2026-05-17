"""Temperature regulator validation script.

Runs a sustained CPU-bound workload and observes the PID-controlled worker pool.
Captures steady-state temperature, overshoot, and settling time.
"""

from __future__ import annotations

import os

TARGET_TEMP_C = float(os.environ.get("TARGET_TEMP_C", "65.0"))
KP = float(os.environ.get("KP", "1.5"))
KI = float(os.environ.get("KI", "0.3"))
MIN_WORKERS = int(os.environ.get("MIN_WORKERS", "4"))


def pid_step(error: float, integral: float) -> tuple[float, float]:
    """Single PID step. Returns (control_output, updated_integral)."""
    integral_next = integral + error
    output = KP * error + KI * integral_next
    return output, integral_next


def simulate_run(duration_s: int = 600) -> dict[str, float]:
    """Simulate a sustained workload and return summary metrics."""
    temp = 50.0
    integral = 0.0
    workers = 8
    overshoot = 0.0
    for _ in range(duration_s):
        error = TARGET_TEMP_C - temp
        delta, integral = pid_step(error, integral)
        workers = max(MIN_WORKERS, int(workers - delta * 0.1))
        temp += workers * 0.05 - 0.4
        overshoot = max(overshoot, temp - TARGET_TEMP_C)
    return {
        "final_temp_c": temp,
        "overshoot_c": overshoot,
        "final_workers": float(workers),
    }


if __name__ == "__main__":
    print(simulate_run())
