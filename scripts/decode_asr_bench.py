"""Decode the locally-downloaded FLEURS vi_vn test parquet into wavs + manifest.

Reads data/asr_bench/fleurs_vi_vn_test.parquet (fetched by hf_hub_download,
resumable) and writes the first N clips as 16 kHz WAVs + manifest.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "asr_bench"
PARQUET = OUT / "fleurs_vi_vn_test.parquet"


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    if not PARQUET.exists():
        print(f"missing {PARQUET} - run the download step first")
        return 1

    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    table = pq.read_table(PARQUET)
    col_names = table.column_names
    audio_col = "audio" if "audio" in col_names else col_names[0]
    raw_col = ("raw_transcription" if "raw_transcription" in col_names
               else "transcription")
    norm_col = "transcription"

    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.jsonl"

    written = 0
    with open(manifest, "w", encoding="utf-8") as fh:
        for i in range(table.num_rows):
            if written >= limit:
                break
            ref = str(table.column(raw_col)[i].as_py() or "").strip()
            if not ref:
                continue
            blob = table.column(audio_col)[i].as_py()
            data = blob["bytes"] if isinstance(blob, dict) else blob
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            fname = f"fleurs_{i:04d}.wav"
            sf.write(str(audio_dir / fname), audio, int(sr))
            fh.write(json.dumps(
                {"file": f"audio/{fname}", "ref": ref}, ensure_ascii=False
            ) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written} clips...")
    print(f"OK -> {written} clips (manifest: {manifest})")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
