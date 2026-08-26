"""Audio IO: push-to-talk capture and the debounced ENTER watcher.

Capture model (operator-friendly for a museum floor):
  * operator presses ENTER  -> recording starts
  * operator presses ENTER again -> recording stops, turn is sent
  * hard cap at ``max_utterance_seconds`` as a safety net

Keyboard polling uses msvcrt on Windows; a no-op fallback keeps imports safe
on other platforms.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from config import SAMPLE_RATE

logger = logging.getLogger("audio")

FRAME_MS = 30
# Conservative RMS bar for trimming leading/trailing quiet off the buffer.
# Toggle mode never gates recording on loudness, so a fixed trim threshold
# is enough - no venue noise-floor calibration needed.
TRIM_RMS = 0.01


class MicRecorder:
    """Streaming mic capture ended ONLY by the operator's ENTER press."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._recording = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("mic status: %s", status)
        if self._recording.is_set():
            with self._lock:
                self._frames.append(indata.copy())

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))

    def _buffer_snapshot(self) -> np.ndarray | None:
        with self._lock:
            if not self._frames:
                return None
            return np.concatenate(self._frames, axis=0).reshape(-1)

    def record_push_to_talk(self, stop_check=None) -> np.ndarray:
        """Pure push-to-toggle: ONLY ``stop_check`` ends the capture.

        ENTER starts (orchestrator), ENTER again stops - a hesitating kid
        can never be cut off by a timer, at the cost of one extra press.
        """
        import sounddevice as sd

        started = time.perf_counter()
        self._frames.clear()
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
            callback=self._callback,
        ):
            self._recording.set()
            logger.info("ENTER pressed - recording started")
            while True:
                time.sleep(FRAME_MS / 1000)
                if stop_check is not None and stop_check():
                    logger.info(
                        "ENTER pressed - recording stopped (%.2fs buffered)",
                        time.perf_counter() - started,
                    )
                    break
                if (
                    time.perf_counter() - started
                ) >= self.cfg.max_utterance_seconds:
                    logger.info("max utterance length reached")
                    break
        self._recording.clear()

        audio = self._concat_frames()
        trimmed = trim_silence(audio, TRIM_RMS, sample_rate=SAMPLE_RATE)
        if trimmed.shape[0] < self.cfg.min_speech_ms / 1000 * SAMPLE_RATE:
            logger.info("captured audio too short; treating as empty")
            return np.zeros(0, dtype=np.float32)
        logger.info(
            "sending %.2fs of speech to ASR", trimmed.shape[0] / SAMPLE_RATE
        )
        return trimmed

    def _concat_frames(self) -> np.ndarray:
        audio = self._buffer_snapshot()
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        self._frames.clear()
        return audio


def trim_silence(
    audio: np.ndarray, threshold: float, pad_ms: int = 80, sample_rate: int = 16000
) -> np.ndarray:
    """Cut leading/trailing near-silence, keep a small padding."""
    if audio.size == 0:
        return audio
    window = int(sample_rate * 0.02)
    n = (audio.shape[0] // window) * window
    if n == 0:
        return audio
    windows = audio[:n].reshape(-1, window)
    active = np.sqrt((windows**2).mean(axis=1)) > threshold
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return audio[:0]
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, indices[0] * window - pad)
    end = min(audio.shape[0], (indices[-1] + 1) * window + pad)
    return audio[start:end]


# ----------------------------------------------------------------------
class EnterKeyWatcher:
    """Background thread watching for ENTER presses (msvcrt-based).

    Consumers either poll ``consume_press()`` or set a callback. On
    non-Windows platforms this degrades gracefully (no key events).
    """

    def __init__(self):
        self._presses = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        try:
            import msvcrt  # noqa: F401 - availability probe

            self._available = True
        except ImportError:
            self._available = False

    def start(self) -> None:
        if not self._available or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="key-watch", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        import msvcrt

        while self._running:
            self._poll_once(msvcrt)
            time.sleep(0.02)

    # Windows key-repeat streams \r events ~30/s while ENTER is held;
    # without debouncing, one hold piles up a backlog of phantom
    # "stop recording" presses that machine-guns the voice loop.
    _DEBOUNCE_S = 0.25

    def _poll_once(self, kb) -> bool:
        """Consume at most one key event. True iff ENTER press ACCEPTED."""
        if not kb.kbhit():
            return False
        key = kb.getwch()  # always consume, even non-ENTER keys
        now = time.monotonic()
        last = getattr(self, "_last_press", 0.0)
        if key not in ("\r", "\n") or now - last < self._DEBOUNCE_S:
            return False
        # A held key streams \r events forever; past the debounce window,
        # only accept when the finger actually came back up in between.
        if self.is_down() and now - last < self._HOLD_SUPPRESS_S:
            return False
        self._last_press = now
        with self._lock:
            self._presses += 1
        return True

    # Beyond this, a still-held key's events are treated as repeats.
    _HOLD_SUPPRESS_S = 30.0

    # ------------------------------------------------------------------
    def is_down(self) -> bool:
        """True while ENTER is physically held (Windows key state)."""
        try:
            import ctypes

            return bool(ctypes.windll.user32.GetAsyncKeyState(0x0D) & 0x8000)
        except Exception:
            return False

    def drain(self) -> None:
        """Drop any buffered presses (e.g. key-repeat backlog)."""
        with self._lock:
            self._presses = 0

    def consume_press(self) -> bool:
        with self._lock:
            if self._presses > 0:
                self._presses -= 1
                return True
            return False
