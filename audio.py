"""Audio IO: push-to-talk capture with silence auto-stop, speaker guard.

Capture model (operator-friendly for a museum floor):
  * operator presses ENTER  -> recording starts
  * turn ends automatically when quiet settles OR the operator presses
    ENTER again, whichever comes first. With Smart Turn enabled
    (smart_turn.py) the quiet pause is classified every ``check_ms``:
    a confident "finished" ends the turn early (~400ms), a mid-sentence
    breath keeps listening up to ``silence_end_ms + max_extra_ms``.
    Without it (or on any failure), the legacy fixed ``silence_end_ms``
    window applies.
  * hard cap at ``max_utterance_seconds``
This removes dead air from the ASR input (the single biggest latency win of
push-to-talk designs) without requiring hands-free VAD.

Keyboard polling uses msvcrt on Windows; a no-op fallback keeps imports safe
on other platforms.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from config import SAMPLE_RATE, SMART_TURN_MODEL

logger = logging.getLogger("audio")

FRAME_MS = 30


class MicRecorder:
    """Streaming mic capture into an in-memory buffer with end-of-turn logic."""

    # Re-estimate the ambient noise floor every N recordings (0.4s of mic
    # time each) instead of before every single turn.
    FLOOR_REFRESH_EVERY = 10

    def __init__(self, cfg, turn_checker=None):
        self.cfg = cfg
        self._stream = None
        self._frames: list[np.ndarray] = []
        self._recording = threading.Event()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._cached_floor: float | None = None
        self._since_floor_check = 0
        # End-of-turn classifier (SmartTurnClassifier). Injected by tests;
        # built lazily from cfg otherwise. None -> legacy fixed window.
        self._smart_turn_enabled = bool(cfg.smart_turn_enabled)
        if self._smart_turn_enabled and turn_checker is not None:
            self._turn_checker = turn_checker
            self._turn_checker_built = True
        else:
            self._turn_checker = None
            self._turn_checker_built = not self._smart_turn_enabled
        self._turn_checker_failed = False

    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("mic status: %s", status)
        if self._recording.is_set():
            with self._lock:
                self._frames.append(indata.copy())

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))

    def _estimate_floor(self) -> float:
        """Sample ambient noise briefly so silence detection adapts to venue."""
        import sounddevice as sd

        duration_s = 0.4
        chunk = sd.rec(
            int(duration_s * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return min(self._rms(chunk), 0.02)

    # ------------------------------------------------------------------
    def _get_turn_checker(self):
        """Resolve the end-of-turn classifier once; None -> legacy window."""
        if self._turn_checker_built or not self._smart_turn_enabled:
            return self._turn_checker
        self._turn_checker_built = True
        try:
            from smart_turn import SmartTurnClassifier

            self._turn_checker = SmartTurnClassifier.create(SMART_TURN_MODEL)
        except Exception as exc:  # pragma: no cover - import-time failure only
            logger.warning("smart-turn import failed (%s) - fixed window stays", exc)
            self._turn_checker = None
        if self._turn_checker is None:
            self._turn_checker_failed = True
        return self._turn_checker

    def _buffer_snapshot(self) -> np.ndarray | None:
        with self._lock:
            if not self._frames:
                return None
            return np.concatenate(self._frames, axis=0).reshape(-1)

    def record_until_turn_end(
        self, floor: float | None = None, progress_cb=None, stop_check=None
    ) -> np.ndarray:
        """Blocks until a full utterance is captured; returns float32 mono.

        ``floor`` pins the silence threshold explicitly (tests); otherwise a
        cached venue estimate is reused and refreshed periodically so each
        ENTER press does not pay 0.4s of calibration again.
        """
        import sounddevice as sd

        if getattr(self.cfg, "ptt_mode", "smart") == "manual":
            return self._record_manual(stop_check=stop_check)

        if floor is None:
            if (
                self._cached_floor is None
                or self._since_floor_check >= self.FLOOR_REFRESH_EVERY
            ):
                self._cached_floor = self._estimate_floor()
                self._since_floor_check = 0
            floor = self._cached_floor
        self._since_floor_check += 1
        threshold = max(floor * 2.5, 0.0035)
        silence_limit_ms = self.cfg.silence_end_ms
        started = time.perf_counter()

        checker = self._get_turn_checker() if not self._turn_checker_failed else None
        check_every_ms = max(self.cfg.smart_turn_check_ms, 100.0)
        # Total quiet tolerated when the classifier keeps saying "mid-sentence".
        max_quiet_ms = silence_limit_ms + self.cfg.smart_turn_max_extra_ms
        turn_threshold = self.cfg.smart_turn_threshold
        # Very short buffers carry too little context to score meaningfully.
        min_classify_samples = int(0.25 * SAMPLE_RATE)
        next_check_ms = check_every_ms

        self._frames.clear()
        self._stop_flag.clear()
        speech_seen_ms = 0.0
        silent_run_ms = 0.0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
            callback=self._callback,
        ):
            self._recording.set()
            logger.info(
                "recording started (noise floor %.4f, threshold %.4f)", floor, threshold
            )
            while not self._stop_flag.is_set():
                time.sleep(FRAME_MS / 1000)
                with self._lock:
                    recent = self._frames[-1] if self._frames else None
                elapsed_ms = (time.perf_counter() - started) * 1000
                level = self._rms(recent) if recent is not None else 0.0
                speaking_now = level > threshold
                if speaking_now:
                    speech_seen_ms += FRAME_MS
                    silent_run_ms = 0.0
                    next_check_ms = check_every_ms
                elif speech_seen_ms > 0:
                    silent_run_ms += FRAME_MS
                if progress_cb:
                    progress_cb(level, speech_seen_ms)
                # Operator stop always wins - even mid-extension.
                if stop_check is not None and stop_check():
                    logger.info("operator ended the turn")
                    break
                if elapsed_ms >= self.cfg.max_utterance_seconds * 1000:
                    logger.info("max utterance length reached")
                    break
                if speech_seen_ms < self.cfg.min_speech_ms or silent_run_ms <= 0:
                    continue
                if checker is None:
                    if silent_run_ms >= silence_limit_ms:
                        logger.info(
                            "auto end-of-turn after %.0fms silence", silent_run_ms
                        )
                        break
                    continue
                if silent_run_ms >= max_quiet_ms:
                    logger.info(
                        "smart-turn budget exhausted - ending after %.0fms quiet",
                        silent_run_ms,
                    )
                    break
                if silent_run_ms < next_check_ms:
                    continue
                snapshot = self._buffer_snapshot()
                if snapshot is None or snapshot.size < min_classify_samples:
                    # Not enough context yet; look again after another interval.
                    next_check_ms = silent_run_ms + check_every_ms
                    continue
                try:
                    prob = checker.predict_end_of_turn(snapshot, SAMPLE_RATE)
                except Exception as exc:
                    # One strike: a broken classifier must never wedge capture.
                    logger.warning(
                        "smart-turn failed (%s); fixed window for rest of session", exc
                    )
                    checker = None
                    self._turn_checker_failed = True
                    continue
                logger.info("smart-turn p=%.2f after %.0fms quiet", prob, silent_run_ms)
                if prob >= turn_threshold:
                    logger.info(
                        "smart end-of-turn after %.0fms quiet (p=%.2f)",
                        silent_run_ms,
                        prob,
                    )
                    break
                # Mid-sentence pause: keep listening, re-check later.
                next_check_ms = silent_run_ms + check_every_ms
            self._recording.clear()

        audio = self._concat_frames()
        trimmed = trim_silence(audio, threshold, sample_rate=SAMPLE_RATE)
        if trimmed.shape[0] < self.cfg.min_speech_ms / 1000 * SAMPLE_RATE:
            logger.info(
                "captured audio too short (%.2fs); treating as empty",
                trimmed.shape[0] / SAMPLE_RATE,
            )
            return np.zeros(0, dtype=np.float32)
        logger.info("captured %.2fs of speech", trimmed.shape[0] / SAMPLE_RATE)
        return trimmed

    def _record_manual(self, stop_check=None) -> np.ndarray:
        """Pure push-to-toggle: ONLY the operator's stop_check ends the turn.

        No silence auto-stop, no classifier - a hesitating kid can never be
        cut off, at the cost of one extra ENTER per turn.
        """
        import sounddevice as sd

        started = time.perf_counter()
        self._frames.clear()
        self._stop_flag.clear()
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
            callback=self._callback,
        ):
            self._recording.set()
            logger.info("recording (manual push-to-talk)...")
            while True:
                time.sleep(FRAME_MS / 1000)
                if stop_check is not None and stop_check():
                    logger.info("operator ended the turn")
                    break
                if (
                    time.perf_counter() - started
                ) >= self.cfg.max_utterance_seconds:
                    logger.info("max utterance length reached")
                    break
        self._recording.clear()

        audio = self._concat_frames()
        # Conservative threshold; no floor calibration delay in this path.
        threshold = max((self._cached_floor or 0.004) * 2.5, 0.0035)
        trimmed = trim_silence(audio, threshold, sample_rate=SAMPLE_RATE)
        if trimmed.shape[0] < self.cfg.min_speech_ms / 1000 * SAMPLE_RATE:
            logger.info("captured audio too short; treating as empty")
            return np.zeros(0, dtype=np.float32)
        return trimmed

    def record_hold(self, down_fn, progress_cb=None) -> np.ndarray:
        """Hold-to-talk: record while ``down_fn()`` is True, release = stop.

        Zero end-of-turn detection latency: the buffered frames already
        contain everything up to the release instant.
        """
        import sounddevice as sd

        floor = self._cached_floor
        if floor is None:
            floor = self._estimate_floor()
            self._cached_floor = floor
        threshold = max(floor * 2.5, 0.0035)

        started = time.perf_counter()
        self._frames.clear()
        self._stop_flag.clear()
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000),
            callback=self._callback,
        ):
            self._recording.set()
            logger.info("recording while ENTER held (hold-to-talk)...")
            time.sleep(FRAME_MS / 1000)
            while True:
                time.sleep(FRAME_MS / 1000)
                held_s = time.perf_counter() - started
                if not down_fn():
                    if held_s * 1000 < self.cfg.min_speech_ms:
                        logger.info("key tap too short - ignoring")
                        return np.zeros(0, dtype=np.float32)
                    logger.info("released after %.2fs", held_s)
                    break
                if progress_cb:
                    recent = self._frames[-1] if self._frames else None
                    progress_cb(
                        self._rms(recent) if recent is not None else 0.0,
                        held_s * 1000,
                    )
                if held_s >= self.cfg.max_utterance_seconds:
                    logger.info("max utterance length reached")
                    break
        self._recording.clear()

        audio = self._concat_frames()
        trimmed = trim_silence(audio, threshold, sample_rate=SAMPLE_RATE)
        if trimmed.shape[0] < self.cfg.min_speech_ms / 1000 * SAMPLE_RATE:
            logger.info("captured audio too short; treating as empty")
            return np.zeros(0, dtype=np.float32)
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
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\r", "\n"):
                    with self._lock:
                        self._presses += 1
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def is_down(self) -> bool:
        """True while ENTER is physically held (Windows key state)."""
        try:
            import ctypes

            return bool(ctypes.windll.user32.GetAsyncKeyState(0x0D) & 0x8000)
        except Exception:
            return False

    def consume_press(self) -> bool:
        with self._lock:
            if self._presses > 0:
                self._presses -= 1
                return True
            return False
