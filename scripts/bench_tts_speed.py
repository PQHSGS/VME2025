"""One-shot TTS speed benchmark: Vieneu threads sweep vs edge-tts.

Measures synthesis wall-time and RTF for a representative kiosk sentence,
so the VIENEU_THREADS / TTS_ENGINE decisions are data-backed.
"""

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

SENTENCE = (
    "Đèn ông sao được làm bằng tre và giấy bóng kính đó cháu, "
    "nhẹ mà rực rỡ sắc màu để mang đi rước đèn đêm Trung thu."
)
SHORT = "Chào cháu,"


def bench_vieneu(threads: int) -> None:
    from vieneu import Vieneu

    t0 = time.perf_counter()
    m = Vieneu(backend="onnx", threads=threads)
    load_s = time.perf_counter() - t0
    # warm
    m.infer(SHORT, voice="Minh Đức")
    for label, text in (("short", SHORT), ("medium", SENTENCE)):
        times = []
        dur = 0.0
        for _ in range(2):
            t1 = time.perf_counter()
            wav = m.infer(text, voice="Minh Đức")
            times.append(time.perf_counter() - t1)
            dur = len(wav) / 48000 if hasattr(wav, "__len__") else 0
        best = min(times)
        print(
            f"vieneu threads={threads or 'auto'} {label:6s}: "
            f"{best*1000:6.0f}ms synth | {dur:4.1f}s audio | RTF {best/max(dur,0.01):4.2f} "
            f"(load {load_s:.0f}s)"
        )
    del m


def bench_edge() -> None:
    import asyncio

    import edge_tts

    async def synth(text: str) -> tuple[float, float]:
        t1 = time.perf_counter()
        first = None
        total = 0
        com = edge_tts.Communicate(text, voice="vi-VN-NamMinhNeural")
        async for chunk in com.stream():
            if chunk.get("type") == "audio":
                if first is None:
                    first = time.perf_counter() - t1
                total += len(chunk["data"])
        return (first or 0) * 1000, (time.perf_counter() - t1) * 1000

    for label, text in (("short", SHORT), ("medium", SENTENCE)):
        first_ms, total_ms = asyncio.run(synth(text))
        print(f"edge-tts           {label:6s}: first-audio {first_ms:5.0f}ms | total {total_ms:5.0f}ms")


if __name__ == "__main__":
    import os

    print(f"cpu_count={os.cpu_count()}")
    bench_edge()
    for th in (0, 2, 4):
        try:
            bench_vieneu(th)
        except Exception as exc:
            print(f"vieneu threads={th}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
