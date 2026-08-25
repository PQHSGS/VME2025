"""Speech-to-text backends.

Backends speak a common protocol:
    transcribe(audio, sample_rate) -> str

``GipformerSTT`` : gipformer-65M int8 via sherpa-onnx (default).
                   7.87% WER on FLEURS-vi, RTF 0.033 (30x realtime on CPU).
``WhisperSTT``    : EraX-WoW-Turbo CT2 via faster-whisper (legacy fallback).
Model loads lazily in a worker thread so startup stays responsive.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger("asr")


def detect_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as exc:  # pragma: no cover
        logger.debug("cuda probe failed: %s", exc)
    return "cpu"


class WhisperSTT:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = detect_device(cfg.asr_device)
        if cfg.asr_compute_type == "auto":
            compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            compute_type = cfg.asr_compute_type
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Blocking load - call once at startup (or inside a thread)."""
        from faster_whisper import WhisperModel  # heavy import

        with self._lock:
            if self._model is not None:
                return
            logger.info(
                "loading ASR %s on %s (%s)...",
                self.cfg.asr_model,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.cfg.asr_model,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cfg.asr_cpu_threads or os.cpu_count() or 4,
            )
            logger.info("ASR ready")

    def load_async(self, callback=None) -> threading.Thread:
        def worker():
            try:
                self.load()
            except Exception:
                logger.exception("ASR load failed")
            finally:
                if callback:
                    callback(self.ready)

        thread = threading.Thread(target=worker, name="asr-load", daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """audio: float32 mono in [-1, 1]. Returns plain text ('' when silent)."""
        if audio.size == 0:
            return ""
        if not self.ready:
            logger.warning("ASR model not loaded yet; dropping %d samples", audio.size)
            return ""
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language=self.cfg.asr_language,
            beam_size=self.cfg.asr_beam_size,
            temperature=0.0,
            vad_filter=self.cfg.asr_vad_filter,
            condition_on_previous_text=self.cfg.asr_condition_on_previous_text,
            hotwords=self.cfg.asr_hotwords or None,
            without_timestamps=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        logger.info("transcribed (%.2fs audio): %r", audio.shape[0] / sample_rate, text)
        return text


class GipformerSTT:
    """gipformer-65M int8 via sherpa-onnx (offline transducer)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._recognizer = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._recognizer is not None

    def load(self) -> None:
        """Blocking load - call once at startup (or inside a thread)."""
        import sherpa_onnx  # lazy heavy import

        from config import GIPFORMER_DIR, SAMPLE_RATE

        with self._lock:
            if self._recognizer is not None:
                return
            tokens = str(GIPFORMER_DIR / "tokens.txt")
            encoder = str(GIPFORMER_DIR / "encoder.int8.onnx")
            decoder = str(GIPFORMER_DIR / "decoder.int8.onnx")
            joiner = str(GIPFORMER_DIR / "joiner.int8.onnx")
            logger.info("loading gipformer ASR (int8) from %s ...", encoder)
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=self.cfg.asr_cpu_threads or os.cpu_count() or 4,
                sample_rate=SAMPLE_RATE,
                debug=False,
            )
            logger.info("ASR ready (gipformer-65M int8)")

    def load_async(self, callback=None) -> threading.Thread:
        def worker():
            try:
                self.load()
            except Exception:
                logger.exception("gipformer ASR load failed")
            finally:
                if callback:
                    callback(self.ready)

        thread = threading.Thread(target=worker, name="asr-load", daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """audio: float32 mono in [-1, 1]. Returns plain text ('' when silent)."""
        if audio.size == 0:
            return ""
        if not self.ready:
            logger.warning("gipformer ASR not loaded yet; dropping %d samples", audio.size)
            return ""

        stream = self._recognizer.create_stream()  # type: ignore[union-attr]
        # sherpa-onnx expects float32 numpy; ensure contiguous.
        audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
        stream.accept_waveform(sample_rate, audio_f32)
        self._recognizer.decode_stream(stream)  # type: ignore[union-attr]
        text = stream.result.text.strip()
        logger.info("transcribed (%.2fs audio): %r", audio.shape[0] / sample_rate, text)
        return text
