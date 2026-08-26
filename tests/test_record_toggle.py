"""Push-to-toggle capture: ONLY the operator's stop_check ends recording.

Uses the stubbed-sounddevice pattern (no mic, compressed real-time).
"""

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


def make_cfg():
    return types.SimpleNamespace(
        min_speech_ms=45.0,
        max_utterance_seconds=5.0,
    )


def push_frame(rec, frame):
    if StubInputStream.active is None:
        return
    with rec._lock:
        rec._frames.append(frame)


def test_quiet_audio_does_not_end_turn(monkeypatch):
    """Continuous QUIET audio must NOT end a turn; only stop_check does."""
    recorder = MicRecorder(make_cfg())
    quiet = np.zeros((FRAME, 1), dtype=np.float32)

    def quiet_feeder():
        for _ in range(40):  # ~1.2s of pure silence
            push_frame(recorder, quiet)
            time.sleep(0.01)

    install_stub_sd(monkeypatch, quiet_feeder)

    start = time.perf_counter()

    def stop_after_250ms():
        return time.perf_counter() - start > 0.25

    audio = recorder.record_push_to_talk(stop_check=stop_after_250ms)
    assert time.perf_counter() - start >= 0.24  # no early silence cut
    assert audio.shape[0] == 0  # pure quiet trims to nothing


def test_stop_check_ends_with_captured_speech(monkeypatch):
    recorder = MicRecorder(make_cfg())
    loud = np.full((FRAME, 1), 0.5, dtype=np.float32)

    def loud_feeder():
        for _ in range(30):  # ~0.9s of speech-level audio
            push_frame(recorder, loud)
            time.sleep(0.01)

    install_stub_sd(monkeypatch, loud_feeder)
    start = time.perf_counter()

    def stop_after_300ms():
        return time.perf_counter() - start > 0.3

    audio = recorder.record_push_to_talk(stop_check=stop_after_300ms)
    assert audio.shape[0] > 0  # loud frames captured and kept after trim


def test_no_frames_returns_empty(monkeypatch):
    """Immediate stop with nothing captured -> empty audio, no crash."""
    recorder = MicRecorder(make_cfg())
    install_stub_sd(monkeypatch, lambda: None)
    audio = recorder.record_push_to_talk(stop_check=lambda: True)
    assert audio.shape[0] == 0
