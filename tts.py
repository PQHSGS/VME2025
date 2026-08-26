"""Streaming TTS with sentence-level pipelining.

Engine chain (see ``build_tts_player``):
  1. VieNeu-TTS v3 Turbo - local, offline, 48 kHz, ONNX/CPU or PyTorch/GPU.
  2. edge-tts - free Microsoft cloud voice (unofficial endpoint; kept as
     automatic fallback when the local engine is unavailable).

Design:
  - ``submit(sentence)`` enqueues text; a synth thread converts it to PCM
    while the previous sentence is still playing (prefetch depth = queue).
  - A playback thread writes decoded int16 frames to the speaker.
  - ``stop()`` is the barge-in path: cancels current synthesis, drops all
    queued audio within milliseconds.
  - LRU cache on normalized text - greetings/fillers repeat constantly;
    ``prewarm`` fills it at startup for fillers/fallback lines.
  - After N consecutive failures the player disables itself so the
    session degrades to text-only instead of hanging the loop.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import OrderedDict, deque
from typing import Callable

import numpy as np

logger = logging.getLogger("tts")

TARGET_RATE = 24000


# ----------------------------------------------------------------------
def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """MP3 bytes -> (int16 mono pcm, sample_rate) via soundfile/libsndfile."""
    import io

    import soundfile as sf

    pcm, sr = sf.read(io.BytesIO(data), dtype="int16")
    if pcm.ndim > 1:  # stereo -> mono
        pcm = pcm.mean(axis=1).astype(np.int16)
    return np.asarray(pcm, dtype=np.int16), int(sr)


def resample_to(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    duration = pcm.shape[0] / src_rate
    target_len = max(1, int(duration * dst_rate))
    x_src = np.linspace(0.0, duration, num=pcm.shape[0], endpoint=False)
    x_dst = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_dst, x_src, pcm.astype(np.float64)).astype(np.int16)


async def _edge_synth(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            buffer.extend(chunk["data"])
    if not buffer:
        raise RuntimeError(f"edge-tts produced no audio for {text!r}")
    return bytes(buffer)


_SENTINEL = object()


class TTSPlayer:
    def __init__(
        self,
        cfg,
        synth_fn: Callable[[str], bytes] | None = None,
        synth_pcm_fn: Callable[[str], tuple[np.ndarray, int]] | None = None,
    ):
        self.cfg = cfg
        self._custom_synth = synth_fn  # returns mp3 bytes (tests/mock)
        self._synth_pcm = synth_pcm_fn  # returns (int16 pcm, rate) - local engines
        if synth_pcm_fn is not None:
            self.engine_name = f"vienneu:{cfg.vienneu_voice}"
            cache_ns = f"vienneu|{cfg.vienneu_voice}|{cfg.vienneu_backend}"
        elif synth_fn is not None:
            self.engine_name = "custom"
            cache_ns = "custom"
        else:
            self.engine_name = "edge-tts"
            cache_ns = f"edge|{cfg.tts_voice}|{cfg.tts_rate}"
        self._text_queue: queue.Queue[tuple[int, str] | object] = queue.Queue()
        self._pcm_queue: queue.Queue[tuple[np.ndarray, int] | object] = queue.Queue(
            maxsize=16
        )
        self._cancel = threading.Event()
        self._stop_workers = threading.Event()
        # Ordered multi-worker synthesis: workers may finish out of order;
        # results file into _done and a sequencer releases them to the play
        # queue strictly in submission order. _gen invalidates in-flight
        # results after a barge-in stop().
        self._seq_lock = threading.Lock()
        self._next_seq = 0
        self._next_out = 0
        self._done: dict[int, tuple[np.ndarray, int] | None] = {}
        self._gen = 0
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._cache_ns = cache_ns
        self.consecutive_failures = 0
        self.disabled = not cfg.tts_enabled
        self._speaking = threading.Event()
        self._out_stream = None
        self._idle_since: float | None = None
        self._last_play_end = 0.0
        self._threads: list[threading.Thread] = []
        # Sentence bookkeeping for barge-in fidelity: what was submitted vs
        # what the visitor actually heard (playback started). Entries are
        # (tag, text) so fillers can be excluded from reply history.
        self._submitted: deque[tuple[str, str]] = deque()
        self._heard: list[tuple[str, str]] = []
        self._book_lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.disabled:
            logger.info("TTS disabled by config - running text-only")
            return
        self._threads = []
        for i in range(max(1, int(self.cfg.tts_synth_workers))):
            t = threading.Thread(
                target=self._synth_worker, name=f"tts-synth-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)
        play = threading.Thread(target=self._play_worker, name="tts-play", daemon=True)
        play.start()
        self._threads.append(play)

    def close(self) -> None:
        self._stop_workers.set()
        self._cancel.set()
        try:
            self._text_queue.put_nowait(_SENTINEL)
            self._pcm_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self._close_output()

    # ------------------------------------------------------------------
    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def busy(self) -> bool:
        # _done must count: between synth finishing and the sequencer
        # releasing it, both queues can be momentarily empty while audio
        # is still pending - wait_done() would otherwise return early.
        with self._seq_lock:
            pending = bool(self._done)
        return (
            self.speaking
            or not self._text_queue.empty()
            or not self._pcm_queue.empty()
            or pending
        )

    def submit(self, sentence: str, tag: str = "reply") -> bool:
        """Queue one sentence. Returns False when TTS is unavailable.

        ``tag`` groups utterances ("reply" vs "filler") for barge-in
        bookkeeping; it does not affect synthesis.
        """
        sentence = sentence.strip()
        if not sentence or self.disabled:
            return False
        self._cancel.clear()
        self._speaking.set()
        with self._book_lock:
            self._submitted.append((tag, sentence))
        with self._seq_lock:
            seq = self._next_seq
            self._next_seq += 1
            gen = self._gen
        self._text_queue.put((seq, sentence, gen))
        return True

    def heard_text(self, tag: str | None = None) -> str:
        """Text the visitor actually heard so far (optionally one tag only)."""
        with self._book_lock:
            texts = [t for k, t in self._heard if tag is None or k == tag]
        return " ".join(texts).strip()

    def reset_reply_bookkeeping(self) -> None:
        """Start a fresh accounting window (call once per bot reply)."""
        with self._book_lock:
            self._submitted.clear()
            self._heard.clear()

    def wait_done(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.busy:
                return True
            time.sleep(0.03)
        return not self.busy

    def stop(self) -> None:
        """Barge-in: drop everything currently queued/playing.

        Sentences whose playback had not started are removed from the
        submitted bookkeeping, so ``heard_text()`` reflects reality.
        """
        drained = 0
        while True:
            try:
                self._text_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        while True:
            try:
                item = self._pcm_queue.get_nowait()
                drained += 1
                if item is not _SENTINEL and item is not None:
                    with self._book_lock:
                        if self._submitted:
                            self._submitted.popleft()
            except queue.Empty:
                break
        self._cancel.set()
        self._speaking.clear()
        with self._seq_lock:
            # Invalidate any in-flight synthesis and reset the ordering
            # state so the next reply starts from a clean sequence.
            self._gen += 1
            self._done.clear()
            self._next_seq = 0
            self._next_out = 0
        with self._book_lock:
            # nothing further will play in this window; keep _heard intact
            self._submitted.clear()
        if drained:
            logger.info("barge-in: dropped %d queued items", drained)

    # ------------------------------------------------------------------
    def _ensure_output(self):
        if self._out_stream is None:
            import sounddevice as sd

            self._out_stream = sd.OutputStream(
                samplerate=TARGET_RATE, channels=1, dtype="int16"
            )
            self._out_stream.start()
        return self._out_stream

    def _close_output(self) -> None:
        stream, self._out_stream = self._out_stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("output stream close failed", exc_info=True)

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        key = f"{self._cache_ns}|{text}"
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if self._synth_pcm is not None:
            pcm, sr = self._synth_pcm(text)
        elif self._custom_synth is not None:
            mp3 = self._custom_synth(text)
            pcm, sr = decode_audio_bytes(mp3)
        else:
            mp3 = asyncio.run(_edge_synth(text, self.cfg.tts_voice, self.cfg.tts_rate))
            pcm, sr = decode_audio_bytes(mp3)
        pcm = resample_to(pcm, sr, TARGET_RATE)
        self._cache[key] = (pcm, TARGET_RATE)
        while len(self._cache) > self.cfg.tts_cache_size:
            self._cache.popitem(last=False)
        return pcm, TARGET_RATE

    def prewarm(self, texts: list[str]) -> int:
        """Synthesize short high-traffic lines into the cache (no playback).

        Called once at warmup for filler phrases and the fallback reply so
        they answer from cache instantly. Failures are logged but never count
        toward the consecutive-failure disable logic.
        """
        done = 0
        for text in texts:
            text = text.strip()
            if not text or self.disabled:
                continue
            try:
                self._synthesize(text)
                done += 1
            except Exception as exc:
                logger.warning("prewarm failed for %r: %s", text[:40], exc)
        return done

    def _file_result(
        self, seq: int, gen: int, result: tuple[np.ndarray, int] | None
    ) -> None:
        """Store a synth outcome and release contiguous results in order.

        ``result=None`` marks a failed synthesis: the slot must still be
        released so later sentences are not stuck behind it. Results from a
        previous generation (pre-barge-in) are discarded.
        """
        with self._seq_lock:
            if gen != self._gen or seq < self._next_out:
                return  # stale: barge-in happened mid-synthesis
            self._done[seq] = result
            while self._next_out in self._done:
                item = self._done.pop(self._next_out)
                self._next_out += 1
                self._pcm_queue.put(item)

    def _synth_worker(self) -> None:
        while not self._stop_workers.is_set():
            try:
                item = self._text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            seq, text, gen = item
            if self._cancel.is_set():
                continue
            try:
                pcm, sr = self._synthesize(text)
                self.consecutive_failures = 0
                self._file_result(seq, gen, (pcm, sr))
            except Exception as exc:
                self._file_result(seq, gen, None)
                self.consecutive_failures += 1
                logger.warning(
                    "TTS synth failed (%d in a row): %s", self.consecutive_failures, exc
                )
                if self.consecutive_failures >= self.cfg.tts_max_consecutive_failures:
                    self.disabled = True
                    logger.error(
                        "TTS disabled after repeated failures - text-only mode"
                    )

    def _play_worker(self) -> None:
        while not self._stop_workers.is_set():
            try:
                item = self._pcm_queue.get(timeout=0.2)
            except queue.Empty:
                if not self._text_queue.empty():
                    continue
                if self._text_queue.empty() and self._pcm_queue.empty():
                    self._maybe_idle()
                continue
            if item is _SENTINEL:
                break
            if item is None:
                # Failed synthesis: keep the submitted/heard bookkeeping
                # aligned by consuming its entry, then move on silently.
                with self._book_lock:
                    if self._submitted:
                        self._submitted.popleft()
                    self._heard.append(("reply", ""))
                continue
            if self._cancel.is_set():
                continue
            pcm, sr = item
            now = time.monotonic()
            gap_s = now - self._last_play_end
            if gap_s > 0.5:
                logger.info("tts: %.2fs audible gap before next sentence", gap_s)
            with self._book_lock:
                if self._submitted:
                    self._heard.append(self._submitted.popleft())
                else:
                    self._heard.append(("reply", ""))
            try:
                stream = self._ensure_output()
                chunk_size = 4800  # ~200ms at 24kHz
                for offset in range(0, pcm.shape[0], chunk_size):
                    if self._cancel.is_set():
                        break
                    stream.write(
                        np.ascontiguousarray(pcm[offset : offset + chunk_size])
                    )
                self._last_play_end = time.monotonic()
            except Exception:
                logger.exception("playback error")
                self._close_output()
            finally:
                if self._text_queue.empty() and self._pcm_queue.empty():
                    self._speaking.clear()

    def _maybe_idle(self) -> None:
        # Release the audio device only after a long quiet period - reopening
        # a WASAPI stream costs ~50-100ms and kiosk turn gaps are usually
        # shorter than this. Tunable via TTS_IDLE_CLOSE_S.
        timeout_s = float(self.cfg.tts_idle_close_s)
        if self._out_stream is not None and not self.busy:
            self._idle_since = self._idle_since or time.time()
            if time.time() - self._idle_since > timeout_s:
                self._close_output()
                self._idle_since = None


# ----------------------------------------------------------------------
def build_tts_player(cfg, probe: bool = False) -> "TTSPlayer":
    """Engine chain: VieNeu-TTS v3 Turbo (local) -> edge-tts (cloud).

    ``probe=True`` synthesizes one tiny line at build time so a broken local
    model (missing weights, bad voice name) is caught during startup instead
    of mid-show. Any VieNeu failure logs a warning and lands on edge-tts.
    """
    if not cfg.tts_enabled:
        logger.info("TTS disabled by config - text-only")
        return TTSPlayer(cfg)

    engine = cfg.tts_engine.lower()
    if engine in ("vienneu", "auto"):
        try:
            from tts_vienneu import VienneuSynth
        except ImportError as exc:
            logger.warning(
                "vieneu SDK unavailable (%s) - falling back to edge-tts", exc
            )
        else:
            if not VienneuSynth.available():
                logger.warning(
                    "vieneu package not importable - falling back to edge-tts"
                )
            else:
                synth = VienneuSynth(
                    cfg.vienneu_voice,
                    cfg.vienneu_backend,
                    threads=cfg.vienneu_threads,
                )
                probed = True
                if probe:
                    try:
                        synth.probe()
                    except Exception as exc:
                        probed = False
                        logger.warning(
                            "VieNeu probe failed (%s) - falling back to "
                            "edge-tts; check VIENEU_VOICE via "
                            "list_preset_voices()",
                            exc,
                        )
                if probed:
                    logger.info(
                        "TTS engine: vienneu v3 turbo (voice=%s backend=%s)",
                        synth.voice,
                        synth.backend,
                    )
                    return TTSPlayer(cfg, synth_pcm_fn=synth)
    elif engine != "edge":
        logger.warning("unknown TTS_ENGINE %r - using edge-tts", engine)
    logger.info("TTS engine: edge-tts (%s %s)", cfg.tts_voice, cfg.tts_rate)
    return TTSPlayer(cfg)
