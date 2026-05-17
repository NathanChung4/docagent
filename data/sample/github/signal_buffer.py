"""Signal buffer validation script.

Drives a producer/consumer pair where the producer is faster than the consumer
and verifies the buffer's drop-policy behavior.
"""

from __future__ import annotations

import os
from collections import deque

CAPACITY = int(os.environ.get("CAPACITY", "4096"))
DROP_POLICY = os.environ.get("DROP_POLICY", "oldest")
FLUSH_INTERVAL_MS = int(os.environ.get("FLUSH_INTERVAL_MS", "50"))


def make_buffer() -> deque[int]:
    """Allocate the underlying ring buffer at the configured capacity."""
    return deque(maxlen=CAPACITY)


def push_with_policy(buf: deque[int], value: int) -> int:
    """Push a value, returning 1 if a previously-buffered entry was dropped."""
    dropped = 0
    if len(buf) == buf.maxlen:
        dropped = 1
        if DROP_POLICY == "newest":
            return dropped  # silently discard the new value
    buf.append(value)
    return dropped


def producer_consumer_run(producer_units: int, consumer_units: int) -> dict[str, int]:
    """Simulate one second of producer/consumer mismatch."""
    buf = make_buffer()
    dropped = 0
    for i in range(producer_units):
        dropped += push_with_policy(buf, i)
    forwarded = min(consumer_units, len(buf))
    return {
        "produced": producer_units,
        "forwarded": forwarded,
        "dropped": dropped,
        "buffer_residual": len(buf) - forwarded,
    }


if __name__ == "__main__":
    print(producer_consumer_run(8000, 4000))
