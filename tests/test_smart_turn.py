"""Offline tests for Smart Turn: classifier wrapper + recorder end-of-turn logic."""

import sys
import threading
import time
import types

import numpy as np
import pytest

from smart_turn import SmartTurnClassifier, compute_whisper_log_mel_features


# ---------------------------------------------------------------------------
# Classifier wrapper (fake ONNX session - no model file, no onnxruntime)
# ---------------------------------------------------------------------------
class FakeSession:
    def __init__(self, probs=None, error: Exception | None = None):
        self.probs = list(probs or [])
        self.error = error
        self.shapes = []

    def run(self, _, inputs):
        self.shapes.append(inputs["input_features"].shape)
        if self.error:
            raise self.error
        p = self.probs.pop(0) if self.probs else 0.9
        return [np.array([[p]], dtype=np.float32)]


def make_classifier(probs=None, error=None) -> SmartTurnClassifier:
    c = object.__new__(SmartTurnClassifier)
    c.model_path = "fake.onnx"
    c._session = FakeSession(probs, error)
    return c


def test_mel_features_shape():
    rng = np.random.default_rng(7)
    feats = compute_whisper_log_mel_features(
        rng.standard_normal(16000).astype(np.float32)
    )
    assert feats.shape == (80, 800)
    assert np.isfinite(feats).all()


def test_predict_clamps_probability():
    hi = make_classifier([1.7])
    lo = make_classifier([-0.5])
    audio = np.zeros(8000, dtype=np.float32)
    assert hi.predict_end_of_turn(audio) == pytest.approx(1.0)
    assert lo.predict_end_of_turn(audio) == pytest.approx(0.0)


def test_predict_truncates_to_8s():
    c = make_classifier([0.8])
    out = c.predict_end_of_turn(np.ones(10 * 16000, dtype=np.float32))
    assert out == pytest.approx(0.8, abs=1e-5)
    assert c._session.shapes[-1] == (1, 80, 800)


def test_create_returns_none_when_model_missing(tmp_path, monkeypatch):
    # Avoid importing onnxruntime in CI-less/offline envs: path check fires first.
    assert SmartTurnClassifier.create(tmp_path / "nope.onnx") is None


# ---------------------------------------------------------------------------
# Recorder end-of-turn logic (stubbed sounddevice, real-time but compressed)
# ---------------------------------------------------------------------------
SR = 16000
FRAME = int(SR * 30 / 1000)


class StubInputStream:
    """Stands in for sounddevice.InputStream; drives a feeder coroutine."""

    active = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        StubInputStream.active = self
        if self.feeder:
            threading.Thread(target=self.feeder, daemon=True).start()
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
        sample_rate=SR,
        silence_end_ms=90.0,
        min_speech_ms=45.0,
        max_utterance_seconds=5.0,
        smart_turn_enabled=True,
        smart_turn_model="fake.onnx",
        smart_turn_threshold=0.5,
        smart_turn_check_ms=90.0,
        smart_turn_max_extra_ms=600.0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def feeder_speaks_then_quiet(recorder, speak_ms=150):
    """Append loud frames for speak_ms, then quiet frames until told to stop."""

    def run():
        rng = np.random.default_rng(3)
        loud = (rng.standard_normal(FRAME) * 0.1).astype(np.float32)
        quiet = np.zeros(FRAME, dtype=np.float32)
        deadline = time.perf_counter() + speak_ms / 1000
        while time.perf_counter() < deadline:
            with recorder._lock:
                recorder._frames.append(loud.copy())
            time.sleep(0.005)
        while StubInputStream.active is not None:
            with recorder._lock:
                recorder._frames.append(quiet.copy())
            time.sleep(0.01)

    return run


def test_smart_turn_extends_pause_then_finishes(monkeypatch):
    from audio import MicRecorder

    cfg = make_cfg()
    checker = make_classifier(probs=[0.1, 0.95])  # pause -> keep listening -> done
    rec = MicRecorder(cfg, turn_checker=checker)
    install_stub_sd(monkeypatch, feeder_speaks_then_quiet(rec))

    audio = rec.record_until_turn_end(floor=0.001)

    assert audio.size > 0
    assert len(checker._session.shapes) == 2  # one extension, one finish


def test_smart_turn_finishes_early_on_confident_pause(monkeypatch, caplog):
    import logging

    from audio import MicRecorder

    cfg = make_cfg()
    checker = make_classifier(probs=[0.95])
    rec = MicRecorder(cfg, turn_checker=checker)
    install_stub_sd(monkeypatch, feeder_speaks_then_quiet(rec, speak_ms=350))

    with caplog.at_level(logging.INFO, logger="audio"):
        audio = rec.record_until_turn_end(floor=0.001)

    assert audio.size > 0
    assert len(checker._session.shapes) == 1
    ends = [
        r.getMessage() for r in caplog.records if "smart end-of-turn" in r.getMessage()
    ]
    assert len(ends) == 1
    quiet_ms = float(ends[0].split("after ")[1].split("ms")[0])
    # Confident finish at/near the first check - far below a realistic window
    # (allow two 30ms loop-tick of scheduling jitter).
    assert quiet_ms <= cfg.silence_end_ms + 60


def test_extra_budget_falls_back_to_fixed_window(monkeypatch):
    from audio import MicRecorder

    # cap = silence_end(90) + extra(300) = 390ms of quiet - enough for >=2
    # classifications despite 30ms loop-tick jitter.
    cfg = make_cfg(smart_turn_max_extra_ms=300.0)
    checker = make_classifier(probs=[0.05] * 50)  # never "finished"
    rec = MicRecorder(cfg, turn_checker=checker)
    install_stub_sd(monkeypatch, feeder_speaks_then_quiet(rec, speak_ms=300))

    audio = rec.record_until_turn_end(floor=0.001)

    assert audio.size > 0
    assert len(checker._session.shapes) >= 2  # did try more than once
    assert len(checker._session.shapes) <= 4  # ...but gave up within budget


def test_runtime_error_disables_checker_for_session(monkeypatch):
    from audio import MicRecorder

    cfg = make_cfg()
    checker = make_classifier(error=RuntimeError("boom"))
    rec = MicRecorder(cfg, turn_checker=checker)
    install_stub_sd(monkeypatch, feeder_speaks_then_quiet(rec))

    audio = rec.record_until_turn_end(floor=0.001)

    assert audio.size > 0
    assert len(checker._session.shapes) == 1  # struck out after first failure
    assert rec._turn_checker_failed


def test_disabled_config_skips_classifier_entirely(monkeypatch):
    from audio import MicRecorder

    cfg = make_cfg(smart_turn_enabled=False)
    sentinel = make_classifier(probs=[0.99])
    rec = MicRecorder(cfg, turn_checker=sentinel)
    install_stub_sd(monkeypatch, feeder_speaks_then_quiet(rec))

    audio = rec.record_until_turn_end(floor=0.001)

    assert audio.size > 0
    assert len(sentinel._session.shapes) == 0  # never consulted
