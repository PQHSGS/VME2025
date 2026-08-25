"""Prepare the Vietnamese ASR benchmark set: FLEURS vi_vn test subset.

Downloads the first N clips of the FLEURS Vietnamese test split (the same
set gipformer and Nemotron-3.5 publish WER numbers on - results are directly
comparable to their tables), saves them as 16 kHz WAVs plus a manifest.

Output:
    data/asr_bench/audio/*.wav
    data/asr_bench/manifest.jsonl   {"file": "...", "ref": "<raw text>"}

FLEURS is CC-BY 4.0 (Google). Run with the .vme_tsg env python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "asr_bench"
LIMIT_DEFAULT = 150


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else LIMIT_DEFAULT
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    from datasets import load_dataset

    print(f"streaming google/fleurs vi_vn test split (first {limit} clips)...")
    ds = load_dataset("google/fleurs", "vi_vn", split="test", streaming=True)

    manifest_path = OUT / "manifest.jsonl"
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if written >= limit:
                break
            ref = (row.get("raw_transcription") or row.get("transcription") or "").strip()
            if not ref:
                continue
            audio = row["audio"]
            fname = f"fleurs_{i:04d}.wav"
            sf.write(
                str(audio_dir / fname),
                audio["array"].astype("float32"),
                int(audio["sampling_rate"]),
            )
            fh.write(json.dumps({"file": f"audio/{fname}", "ref": ref},
                                ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written} clips...")
    print(f"OK -> {written} clips under {audio_dir} (manifest: {manifest_path})")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
