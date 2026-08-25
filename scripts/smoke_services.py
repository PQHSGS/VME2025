"""End-to-end microservice smoke test (real processes, real models).

Boots a subset of services via ServiceManager, then exercises each remote
client against them. Use after changing any service or client code:

    python scripts/smoke_services.py                # rag + tts + llm
    python scripts/smoke_services.py --only rag     # lighter subset

Exits non-zero on the first failed stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Microservice smoke test")
    parser.add_argument(
        "--only", default="rag,tts,llm", help="comma list: rag|tts|llm|asr"
    )
    args = parser.parse_args()
    names = [p.strip() for p in args.only.split(",") if p.strip()]

    from config import Config, setup_logging

    cfg = Config()
    setup_logging(cfg)

    from services.manager import ServiceManager

    manager = ServiceManager(enable_reload=False)
    try:
        manager.start_all(only=names)
        failed = [n for n in names if not manager.services[n].ready]
        print(f"[smoke] ready: {sorted(set(names) - set(failed))} | FAIL: {failed}")
        if "rag" in failed or "tts" in failed:
            print("[smoke] core services not ready - aborting")
            return 1

        from services.clients import (
            RemoteEmbedder,
            RemoteLLM,
            RemoteRetriever,
            RemoteSTT,
            RemoteSituations,
            RemoteSynth,
            build_remote_tts_player,
        )

        ok = True
        if "rag" in names and "rag" not in failed:
            retriever = RemoteRetriever(cfg)
            vec = retriever.embedder.encode_query("Đèn ông sao làm từ gì?")
            print(f"[smoke] embed dim={vec.shape} norm={np.linalg.norm(vec):.3f}")
            result = retriever.retrieve("Đèn ông sao làm từ gì?", q_vec=vec)
            print(f"[smoke] retrieve docs={len(result.docs)} query_used={result.query_used!r}")
            ok &= len(result.docs) > 0
            situations = RemoteSituations(cfg)
            hit = situations.match("chào ông")
            print(f"[smoke] situation hit={hit is not None}")
        if "tts" in names and "tts" not in failed:
            player = build_remote_tts_player(cfg)
            pcm, rate = RemoteSynth(cfg)("Xin chào các em nhỏ!")
            print(f"[smoke] synth {pcm.size} samples @ {rate}Hz dtype={pcm.dtype}")
            ok &= rate == 24000 and pcm.dtype == np.int16 and pcm.size > 0
        if "llm" in names and "llm" not in failed:
            llm = RemoteLLM(cfg)
            text = "".join(llm.stream([{"role": "user", "content": "trả lời đúng một chữ: ok"}]))
            print(f"[smoke] llm stream -> {text[:60]!r}")
            ok &= len(text) > 0
        if "asr" in names:
            stt = RemoteSTT(cfg)
            print(f"[smoke] asr ready={stt.ready}")
            if stt.ready:
                import soundfile as sf

                wav = next(iter((Path("data") / "asr_bench" / "audio").glob("*.wav")), None)
                if wav:
                    pcm, sr = sf.read(wav, dtype="float32")
                    text = stt.transcribe(pcm[: sr * 5], sr)
                    print(f"[smoke] asr sample -> {text!r}")
                    ok &= len(text) > 0
                else:
                    print("[smoke] no bench wav found - skip live decode")

        print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        manager.stop_all()


if __name__ == "__main__":
    sys.exit(main())
