"""Vietnamese ASR bake-off: EraX-WoW-Turbo vs gipformer-65M vs Nemotron-3.5.

Runs every selected backend over data/asr_bench/manifest.jsonl (FLEURS vi_vn
subset) with ONE shared normalization, reporting diacritics-stripped WER
(gipformer-paper style), Vietnamese-diacritics-kept WER, and RTF.

Usage:
    python scripts/bench_asr.py --models erax,gipformer-i8,nemotron --limit 150

Decision rule (AGENTS.md): swap only if a challenger beats EraX by >=2pts
WER at comparable CPU cost - or wins on streaming capability within ~1pt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "asr_bench"
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize(text: str, strip_diacritics: bool) -> str:
    t = text.lower().strip()
    t = PUNCT_RE.sub(" ", t)
    if strip_diacritics:
        t = "".join(c for c in unicodedata.normalize("NFD", t)
                    if not unicodedata.combining(c)).replace("đ", "d")
    return SPACE_RE.sub(" ", t).strip()


# ---------------------------------------------------------------------------
class EraxBackend:
    name = "erax-i8"

    def load(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
            device="cpu", compute_type="int8", cpu_threads=4,
        )

    def transcribe(self, samples: "np.ndarray") -> str:  # noqa: F821
        segments, _ = self._model.transcribe(
            samples, language="vi", beam_size=1, temperature=0.0,
            vad_filter=False, condition_on_previous_text=False,
            without_timestamps=True,
        )
        return "".join(s.text for s in segments).strip()


class GipformerBackend:
    def __init__(self, quantize: str):
        self.name = f"gipformer-{quantize}"

    def load(self):
        import sherpa_onnx
        from huggingface_hub import hf_hub_download

        repo = "g-group-ai-lab/gipformer-65M-rnnt"
        quant = "int8" if self.name.endswith("i8") else "fp32"
        suffix = ".int8.onnx" if quant == "int8" else ".onnx"
        paths = {k: hf_hub_download(repo_id=repo, filename=f"{k}{suffix}")
                 for k in ("encoder", "decoder", "joiner")}
        paths["tokens"] = hf_hub_download(repo_id=repo, filename="tokens.txt")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            num_threads=4, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", **paths,
        )

    def transcribe(self, samples: "np.ndarray") -> str:  # noqa: F821
        stream = self._rec.create_stream()
        stream.accept_waveform(16000, samples)
        self._rec.decode_streams([stream])
        return stream.result.text.strip()


class NemotronBackend:
    name = "nemotron-0.6b"

    def load(self):
        import torch
        from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

        mid = "nvidia/nemotron-3.5-asr-streaming-0.6b"
        self._proc = AutoProcessor.from_pretrained(mid)
        self._model = Nemotron3_5AsrForRNNT.from_pretrained(mid).eval()

    def transcribe(self, samples: "np.ndarray") -> str:  # noqa: F821
        import torch

        inputs = self._proc(samples, sampling_rate=16000,
                            language="vi-VN", return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(**inputs)
        return self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


BACKENDS = {
    "erax": EraxBackend,
    "gipformer-i8": lambda: GipformerBackend("int8"),
    "gipformer-fp32": lambda: GipformerBackend("fp32"),
    "nemotron": NemotronBackend,
}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="erax,gipformer-i8",
                    help=",".join(BACKENDS))
    ap.add_argument("--limit", type=int, default=0, help="cap clips (0 = all)")
    args = ap.parse_args()

    manifest_path = DATA / "manifest.jsonl"
    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} clips from {manifest_path}")

    import numpy as np
    import soundfile as sf

    results = []
    for backend_name in args.models.split(","):
        backend_name = backend_name.strip()
        if backend_name not in BACKENDS:
            print(f"!! unknown backend {backend_name}")
            continue
        backend = BACKENDS[backend_name]()
        try:
            backend.load()
        except Exception as exc:
            print(f"!! {backend_name}: load failed: {exc}")
            continue

        total_audio_s = 0.0
        total_infer_s = 0.0
        errs_nd, errs_vi, words_total = 0.0, 0.0, 0
        try:
            from jiwer import wer as jiwer_wer
        except ImportError:
            print("!! pip install jiwer")
            return 1

        for i, row in enumerate(rows):
            samples, sr = sf.read(str(DATA / row["file"]), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            total_audio_s += len(samples) / sr
            t0 = time.perf_counter()
            try:
                hyp = backend.transcribe(samples)
            except Exception:
                hyp = ""
                print(f"   [{backend_name}] clip {i} failed")
            total_infer_s += time.perf_counter() - t0

            ref_nd = normalize(row["ref"], strip_diacritics=True)
            hyp_nd = normalize(hyp, strip_diacritics=True)
            ref_vi = normalize(row["ref"], strip_diacritics=False)
            hyp_vi = normalize(hyp, strip_diacritics=False)
            n_words = max(len(ref_nd.split()), 1)
            words_total += n_words
            errs_nd += jiwer_wer(ref_nd, hyp_nd) * n_words
            errs_vi += jiwer_wer(ref_vi, hyp_vi) * n_words

        rtf = total_infer_s / max(total_audio_s, 1e-6)
        res = {
            "backend": backend.name,
            "wer_normalized": round(100 * errs_nd / words_total, 2),
            "wer_diacritics_kept": round(100 * errs_vi / words_total, 2),
            "rtf_cpu": round(rtf, 3),
            "audio_s": round(total_audio_s, 1),
        }
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    out = ROOT / "logs" / f"asr_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
