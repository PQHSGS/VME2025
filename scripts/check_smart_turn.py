"""Verify smart_turn against the real ONNX model + pipecat reference impl.

1. Our vendored log-mel must match pipecat's bit-for-bit.
2. Real inference: sane probabilities + latency budget on CPU.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(
    0,
    str(
        ROOT.parent
        / "references"
        / "pipecat"
        / "src"
        / "pipecat"
        / "audio"
        / "turn"
        / "smart_turn"
    ),
)

import numpy as np  # noqa: E402 - path bootstrap above

from smart_turn import SmartTurnClassifier, compute_whisper_log_mel_features as ours  # noqa: E402
from _whisper_features import compute_whisper_log_mel_features as theirs  # noqa: E402

rng = np.random.default_rng(11)
for n in (1600, 16000, 128000):
    a = rng.standard_normal(n).astype(np.float32) * 0.05
    diff = float(np.abs(ours(a) - theirs(a)).max())
    print(f"mel parity n={n}: max|diff|={diff:.2e}")
    assert diff == 0.0, "vendored mel diverges from pipecat"

clf = SmartTurnClassifier.create(ROOT / "models" / "smart-turn-v3.2-cpu.onnx")
assert clf is not None

# Warm-up call absorbs session init (~300ms); measure steady-state only.
clf.predict_end_of_turn(np.zeros(8000, dtype=np.float32))

sr = 16000
t = np.arange(4 * sr) / sr
speech_like = 0.15 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # sustained tone
silence = np.zeros(4 * sr, dtype=np.float32)

cases = {
    "tone-then-silence": np.concatenate(
        [speech_like[: int(2 * sr)], silence[: int(2 * sr)]]
    ),
    "pure silence": silence,
    "continuous tone": speech_like,
}
for name, buf in cases.items():
    t0 = time.perf_counter()
    p = clf.predict_end_of_turn(buf)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{name:>20}: p={p:.3f} ({ms:.0f} ms)")
    assert 0.0 <= p <= 1.0
    assert ms < 250, f"classification too slow: {ms:.0f}ms"
print("OK")
