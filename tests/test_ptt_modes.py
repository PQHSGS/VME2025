"""Tests for PTT_MODE branches: manual (only stop_check ends) and hold
(release ends instantly). Uses the stubbed sounddevice pattern from
test_smart_turn.py — no mic, compressed real-time."""

import sys
import threading
import time
import types

import numpy as np

from audio import MicRecorder

SR = 16000
FRAME = int(SR * 30 / 1000)


class StubInputStream:
    active = None
    feeder = None
    recorder = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        StubInputStream.active = self
        if StubInputStream.feeder:
            threading.Thread(target=StubInputStream.feeder, daemon=True).start()
        return self

    def __exit__(self, *exc):
        StubInputStream.active = None
        return False


def install_stub_sd(monkeypatch, feeder):
    StubInputStream.feeder = staticmethod(feeder)
    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(InputStream=StubInputStream)
    )


def make_cfg(**over):
    base = dict(
        silence_end_ms=90.0,
        min_speech_ms=45.0,
        max_utterance_seconds=5.0,
        smart_turn_enabled=False,
        ptt_mode="smart",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def push_frame(rec, frame):
    if StubInputStream.active is None:
        return
    with rec._lock:
        rec._frames.append(frame)


def test_manual_mode_ignores_silence(monkeypatch):
    """Continuous QUIET audio must NOT end a manual turn; only stop_check."""
    recorder = MicRecorder(make_cfg(ptt_mode="manual"))
    StubInputStream.recorder = recorder
    quiet = np.zeros((FRAME, 1), dtype=np.float32)

    def quiet_feeder():
        for _ in range(40):  # ~1.2s of pure silence
            push_frame(recorder, quiet)
            time.sleep(0.01)

    install_stub_sd(monkeypatch, quiet_feeder)
    recorder._cached_floor = 0.001

    release_at = {"t": None}
    start = time.perf_counter()

    def stop_after_250ms():
        if release_at["t"] is None and time.perf_counter() - start > 0.25:
            release_at["t"] = time.perf_counter() - start
            return True
        return False

    audio = recorder.record_until_turn_end(stop_check=stop_after_250ms)
    held_for = release_at["t"]
    # A smart/silence mode would have cut at ~90-120ms of quiet; manual kept
    # recording until WE said stop.
    assert held_for >= 0.24
    assert audio.shape[0] == 0  # pure quiet trims to nothing


def test_hold_mode_stops_on_release(monkeypatch):
    recorder = MicRecorder(make_cfg())
    StubInputStream.recorder = recorder
    loud = np.full((FRAME, 1), 0.5, dtype=np.float32)

    def loud_feeder():
        for _ in range(60):
            push_frame(recorder, loud)
            time.sleep(0.01)

    install_stub_sd(monkeypatch, loud_feeder)
    recorder._cached_floor = 0.001
    down = {"state": True}

    def releaser():
        time.sleep(0.15)
        down["state"] = False

    threading.Thread(target=releaser, daemon=True).start()
    t0 = time.perf_counter()
    audio = recorder.record_hold(down_fn=lambda: down["state"])
    elapsed = time.perf_counter() - t0

    # Release at ~150ms -> returns promptly after (frame granularity), not
    # after any silence window.
    assert 0.13 <= elapsed <= 1.2
    assert audio.shape[0] > 0  # loud frames were captured


def test_hold_mode_tap_too_short_returns_empty(monkeypatch):
    """A stray key tap captures nothing meaningful -> empty audio."""
    recorder = MicRecorder(make_cfg())
    StubInputStream.recorder = recorder
    install_stub_sd(monkeypatch, lambda: None)  # silence - nothing captured
    recorder._cached_floor = 0.001
    audio = recorder.record_hold(down_fn=lambda: False)  # never held
    assert audio.shape[0] == 0
