import logging
import time

import numpy as np
import pytest

from resilience import Deadline, FailureTracker, budget


def test_deadline_flow():
    d = Deadline(0.05)
    assert not d.expired and d.remaining > 0
    time.sleep(0.06)
    assert d.expired
    with pytest.raises(TimeoutError):
        d.check("unit")


def test_budget_logs_warning(caplog):
    with caplog.at_level(logging.DEBUG, logger="resilience"):
        with budget("fast.block", 5.0):
            pass
        with budget("slow.block", 0.01):
            time.sleep(0.03)
    msgs = [r.message for r in caplog.records]
    assert any("fast.block" in m for m in msgs)
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("slow.block" in r.message for r in warn)


def test_failure_tracker_crosses_once():
    t = FailureTracker("dep", threshold=3)
    assert [t.record_failure() for _ in range(4)] == [False, False, True, False]
    t.record_success()
    assert t.count == 0


def test_numpy_import_guard():
    # resilience must stay dependency-free; numpy here only guards env sanity
    assert np is not None
