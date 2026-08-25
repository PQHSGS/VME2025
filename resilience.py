"""Latency budgets, deadlines and small resilience helpers.

The orchestrator gives every stage a soft budget (log a warning when breached)
and the whole turn a hard deadline (abort gracefully). ``FailureTracker``
powers the LLM circuit breaker. Keeping this logic in one place makes the
latency contract explicit and tunable.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("resilience")


class Deadline:
    """Absolute wall-clock deadline with remaining() convenience."""

    def __init__(self, seconds: float, started_at: float | None = None):
        self.started_at = time.perf_counter() if started_at is None else started_at
        self.expires_at = self.started_at + seconds

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.perf_counter())

    @property
    def expired(self) -> bool:
        return time.perf_counter() >= self.expires_at

    def check(self, what: str = "deadline") -> None:
        if self.expired:
            raise TimeoutError(f"{what} exceeded ({self.elapsed:.2f}s)")


@contextmanager
def budget(name: str, soft_seconds: float):
    """Log a warning when the wrapped block exceeds `soft_seconds`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        level = logging.WARNING if elapsed > soft_seconds else logging.DEBUG
        logger.log(
            level, "[budget] %s took %.3fs (soft %.3fs)", name, elapsed, soft_seconds
        )


@dataclass
class FailureTracker:
    """Counts consecutive failures of an external dependency."""

    name: str
    threshold: int = 3
    count: int = field(default=0, repr=False)

    def record_success(self) -> None:
        self.count = 0

    def record_failure(self) -> bool:
        """Returns True exactly once when the threshold is crossed."""
        self.count += 1
        if self.count == self.threshold:
            logger.error("%s failed %d times in a row", self.name, self.count)
            return True
        return False
