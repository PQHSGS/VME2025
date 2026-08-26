"""Generate A/B TTS samples: sentence-streamed vs whole-line synthesis.

Uses the PRODUCTION SentenceSplitter (same chunk boundaries the kiosk
speaks) and the real VieNeu engine, then writes:

    logs/tts_samples/full_<n>.wav        whole reply in one synth call
    logs/tts_samples/streamed_<n>.wav    per-sentence chunks concatenated

Listen to both and judge whether sentence joins sound awkward.
Also prints per-chunk synth latency (the actual TTFA driver).

    python scripts/tts_samples.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# UTF-8 console before any project import can print Vietnamese.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402 - needs the sys.path shim above
from sentences import SentenceSplitter  # noqa: E402

SAMPLE_REPLIES = [
    (
        "Chào cháu, ông là Tiến sĩ giấy đây. Đèn ông sao được làm bằng tre và "
        "giấy bóng kính đó cháu, nhẹ nhàng mà rực rỡ sắc màu để các cháu mang "
        "đi rước đèn đêm Trung thu. Cháu thấy chiếc đèn ông sao năm nay có "
        "đẹp không?"
    ),
    (
        "À chuyện Chú Cuội hay lắm nè. Chú Cuội ngồi dưới gốc cây đa, cây đa "
        "ấy mãi mãi không già đâu. Vì thế mà đêm rằm ông bay lên cung trăng "
        "ngồi cạnh chị Hằng luôn đó cháu."
    ),
]

OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "tts_samples"


def main() -> int:
    from tts import resample_to
    from tts_vienneu import VienneuSynth

    cfg = Config()
    synth = VienneuSynth(cfg.vienneu_voice, cfg.vienneu_backend)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for n, reply in enumerate(SAMPLE_REPLIES, 1):
        print(f"\n=== sample {n}: {reply[:50]}... ===")

        started = time.perf_counter()
        full_pcm, full_rate = synth(reply)
        full_ms = (time.perf_counter() - started) * 1000
        print(f"full   : {len(full_pcm)/24000:.1f}s audio, synth {full_ms:.0f}ms")

        splitter = SentenceSplitter(early_first_clause=True)
        chunks = splitter.push(reply) + splitter.flush()
        pieces: list[np.ndarray] = []
        total_ms = 0.0
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            pcm, rate = synth(chunk)
            ms = (time.perf_counter() - t0) * 1000
            total_ms += ms
            pcm24 = resample_to(pcm, rate, 24000)
            pieces.append(pcm24)
            print(
                f"chunk {i}: {len(pcm24)/24000:.1f}s synth {ms:.0f}ms | {chunk[:40]!r}"
            )
        streamed_pcm = np.concatenate(pieces) if pieces else np.zeros(0, np.int16)

        sf.write(OUT_DIR / f"full_{n}.wav", full_pcm, full_rate, subtype="PCM_16")
        sf.write(OUT_DIR / f"streamed_{n}.wav", streamed_pcm, 24000, subtype="PCM_16")
        print(
            f"streamed total synth {total_ms:.0f}ms vs full {full_ms:.0f}ms "
            f"-> TTFA win = first-chunk latency only"
        )

    print(f"\nwrote {OUT_DIR}\\full_*.wav and streamed_*.wav - listen and compare joins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
