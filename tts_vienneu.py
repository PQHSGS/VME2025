"""VieNeu-TTS v3 Turbo synthesis backend - local, offline, 48 kHz.

Wraps the ``vieneu`` SDK (``pip install vieneu``): torch-free ONNX runtime
on CPU (int8 quantized), auto-PyTorch on CUDA machines. Returns
``(int16 pcm, sample_rate)`` for TTSPlayer's resample/play pipeline. Any
failure raises so the caller can fall back to edge-tts.

Voice presets: ``Vieneu().list_preset_voices()`` lists ~20 curated voices
across Bac/Trung/Nam regions. Configure via ``VIENEU_VOICE``.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger("tts.vienneu")

VIENEU_RATE = 48000


class VienneuSynth:
    """Callable synthesizer: text -> (int16 mono pcm @ 48kHz, 48000)."""

    def __init__(self, voice: str = "", backend: str = "onnx", threads: int = 0):
        # "Adam" is the SDK's documented default preset; used when the
        # configured voice name does not match any preset.
        self.voice = voice.strip() or "Adam"
        self.backend = backend
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from vieneu import Vieneu

                    logger.info(
                        "loading VieNeu-TTS v3 Turbo (backend=%s voice=%s "
                        "threads=%s)",
                        self.backend,
                        self.voice,
                        self.threads or "auto",
                    )
                    self._model = Vieneu(backend=self.backend, threads=self.threads or 0)
                    self._validate_voice()
        return self._model

    def _validate_voice(self) -> None:
        try:
            presets = [vid for _, vid in self._model.list_preset_voices()]
        except Exception:
            return
        if presets and self.voice not in presets:
            logger.warning(
                "voice %r not among %d presets - SDK may error; check "
                "list_preset_voices()",
                self.voice,
                len(presets),
            )

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        model = self._ensure()
        audio = model.infer(text, voice=self.voice)
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        pcm16 = np.clip(pcm * 32767.0, -32768.0, 32767.0).astype(np.int16)
        return pcm16, VIENEU_RATE

    @staticmethod
    def available() -> bool:
        try:
            import vieneu  # noqa: F401

            return True
        except Exception:
            return False

    def probe(self) -> bool:
        """Load weights and synthesize one short line; raises on failure."""
        self("Xin chào các em nhỏ!")
        return True
