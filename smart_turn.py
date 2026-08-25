"""Smart Turn v3.2 - learned end-of-turn detection (ONNX, CPU, ~10ms).

Replaces the fixed ``silence_end_ms`` tail with a classifier: when the mic
goes quiet, the buffered audio is scored for "did the visitor finish their
question?". A pause mid-sentence ("cho em hỏi... <breath>") scores low and we
keep listening; a finished question scores high and the turn ends after a
much shorter quiet window than a one-size-fits-all timer needs.

Model: smart-turn-v3.2-cpu.onnx (~8MB), bundled from pipecat
(BSD-2-Clause, Daily) which in turn distributes LiveKit/Daily's smart-turn.
The Whisper log-mel feature extractor below is vendored from pipecat's
numpy implementation so inference needs only onnxruntime + numpy.

Any failure (missing model, onnxruntime error) degrades to the legacy fixed
silence window - callers treat this module as optional.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger("audio.smart_turn")

MODEL_RATE = 16000

# ---------------------------------------------------------------------------
# Vendored Whisper-style log-mel features (pipecat, BSD-2-Clause).
# Mirrors transformers.WhisperFeatureExtractor(chunk_length=8).
# ---------------------------------------------------------------------------
_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 80
_SAMPLING_RATE = 16000
_MEL_FLOOR = 1e-10
_NORM_VARIANCE_EPS = 1e-7


def _hertz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(
        logstep * (mels[log_region] - min_log_mel)
    )
    return freq


def _build_mel_filterbank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    mel_min = float(
        _hertz_to_mel_slaney(np.array([min_frequency], dtype=np.float64))[0]
    )
    mel_max = float(
        _hertz_to_mel_slaney(np.array([max_frequency], dtype=np.float64))[0]
    )
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (
        filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters]
    )
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


def _periodic_hann_window(window_length: int) -> np.ndarray:
    return np.hanning(window_length + 1)[:-1]


_HANN_WINDOW = _periodic_hann_window(_N_FFT)
_MEL_FILTERS = _build_mel_filterbank(
    num_frequency_bins=_N_FFT // 2 + 1,
    num_mel_filters=_N_MELS,
    min_frequency=0.0,
    max_frequency=_SAMPLING_RATE / 2.0,
    sampling_rate=_SAMPLING_RATE,
)


def compute_whisper_log_mel_features(
    audio: np.ndarray, *, do_normalize: bool = True
) -> np.ndarray:
    """Whisper-style (80, 800) log-mel features from <=8s of 16 kHz audio."""
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")

    x = np.asarray(audio, dtype=np.float32)
    n_samples = _SAMPLING_RATE * 8  # 128000
    if x.size < n_samples:
        x = np.pad(x, (0, n_samples - x.size), mode="constant")
    elif x.size > n_samples:
        x = x[:n_samples]

    if do_normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + _NORM_VARIANCE_EPS)

    pad = _N_FFT // 2
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    windows = sliding_window_view(padded, _N_FFT)[::_HOP_LENGTH]
    spec = np.fft.rfft(windows * _HANN_WINDOW.astype(np.float64), axis=-1)
    magnitudes = (np.abs(spec) ** 2).T

    mel_spec = np.maximum(_MEL_FLOOR, _MEL_FILTERS.T @ magnitudes)
    log_spec = np.log10(mel_spec)[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------
class SmartTurnClassifier:
    """Lazy ONNX session around smart-turn v3.2; probability in [0, 1]."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._session = None
        import onnxruntime as ort  # moderate import; keep off the hot import path

        options = ort.SessionOptions()
        # One thread: the kiosk box shares CPU with ASR/embedder/TTS.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def create(cls, model_path: str | Path) -> "SmartTurnClassifier | None":
        """Build a classifier, or None when unavailable (missing model/pkg)."""
        try:
            if not Path(model_path).exists():
                logger.info(
                    "smart-turn model not found at %s - fixed window stays", model_path
                )
                return None
            return cls(model_path)
        except Exception as exc:
            logger.warning(
                "smart-turn unavailable (%s) - fixed silence window stays", exc
            )
            return None

    def predict_end_of_turn(
        self, audio: np.ndarray, sample_rate: int = MODEL_RATE
    ) -> float:
        """P(visitor finished their turn); audio float32 mono @ sample_rate."""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != MODEL_RATE:
            try:
                import soxr

                x = soxr.resample(x, sample_rate, MODEL_RATE, quality="HQ")
            except ImportError:
                # Linear fallback is fine for near-16k capture rates.
                n_out = int(round(x.size * MODEL_RATE / sample_rate))
                x = np.interp(
                    np.linspace(0.0, x.size - 1.0, n_out, dtype=np.float64),
                    np.arange(x.size, dtype=np.float64),
                    x.astype(np.float64),
                ).astype(np.float32)

        # Keep the last 8 seconds (the model's full context window).
        max_samples = 8 * MODEL_RATE
        if x.size > max_samples:
            x = x[-max_samples:]

        log_mel = compute_whisper_log_mel_features(x, do_normalize=True)
        outputs = self._session.run(
            None, {"input_features": np.expand_dims(log_mel, axis=0)}
        )
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])
        return min(max(prob, 0.0), 1.0)
