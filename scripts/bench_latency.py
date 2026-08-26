"""Latency benchmark: per-stage timings over the live brain (no mic needed).

Runs N turns through ConversationOrchestrator with the configured LLM backend
(mock by default) and reports stage percentiles from the same instrumentation
used in production traces.

Usage:
    python scripts/bench_latency.py                 # mock backend, 20 turns
    python scripts/bench_latency.py --backend gemini --turns 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "đèn ông sao làm bằng gì vậy ông?",
    "kể cho cháu nghe sự tích chú Cuội đi",
    "bánh nướng khác bánh dẻo thế nào ạ",
    "múa lân là gì vậy ông ơi",
    "ông bao nhiêu tuổi rồi?",
    "cháu thích múa lân lắm!",
]


class RecorderTTS:
    """Captures what would be spoken; measures first-audio submission."""

    def __init__(self):
        self.submitted: list[tuple[str, str]] = []
        self.disabled = False
        self.first_submit_at = None

    def submit(self, sentence, tag="reply"):
        if self.first_submit_at is None:
            self.first_submit_at = time.perf_counter()
        self.submitted.append((tag, sentence))
        return True

    busy = False
    speaking = False

    def wait_done(self, timeout=5.0):
        return True

    def reset_reply_bookkeeping(self):
        pass

    def heard_text(self, tag=None):
        return ""

    def start(self):
        pass

    def close(self):
        pass

    def stop(self):
        pass

    def prewarm(self, phrases):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument(
        "--backend", default=None, help="override LLM_BACKEND for this run"
    )
    args = parser.parse_args()

    from config import Config
    from memory import MemoryManager
    from orchestrator import ConversationOrchestrator

    cfg = Config()
    cfg.telemetry_enabled = False
    if args.backend:
        cfg.llm_backend = args.backend

    tts = RecorderTTS()
    orch = ConversationOrchestrator(
        cfg,
        retriever=None,
        situations=None,
        memory_manager=MemoryManager(cfg),
        tts=tts,
        stt=None,
    )
    from llm import select_backend

    orch.llm = select_backend(cfg)
    orch.retriever = None  # latency of pure brain path; RAG has own bench
    print(f"backend={orch.llm.name} turns={args.turns}\n")

    ttfts, totals, first_audio = [], [], []
    for i in range(args.turns):
        question = QUESTIONS[i % len(QUESTIONS)]
        tts.first_submit_at = None
        started = time.perf_counter()
        reply = orch.process_text(question)
        total = time.perf_counter() - started
        if tts.first_submit_at is not None:
            first_audio.append(tts.first_submit_at - started)
        # re-derive TTFT from trace marks is overkill here; approximate with
        # first sentence submission minus a synth-free pipeline (== llf TTFT)
        ttfts.append(first_audio[-1] if first_audio else total)
        totals.append(total)
        print(f"  [{i + 1:>2}] {total:5.2f}s  q={question[:36]!r} r={reply[:40]!r}")

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p / 100 * (len(xs) - 1)))]

    print("\n--- summary ---")
    print(
        f"first-audio p50={statistics.median(first_audio):.2f}s "
        f"p95={pct(first_audio, 95):.2f}s"
        if first_audio
        else "no audio"
    )
    print(f"turn-total p50={statistics.median(totals):.2f}s p95={pct(totals, 95):.2f}s")
    target_s = 1.6
    ok = first_audio and pct(first_audio, 95) < target_s
    print(f"TTFA p95 target <{target_s}s: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
