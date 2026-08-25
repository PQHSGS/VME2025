"""Summarize logs/traces.jsonl - the offline half of the tuning loop.

Usage:
    python scripts/trace_summary.py                 # today's file
    python scripts/trace_summary.py path/to.jsonl

Prints turn counts by path, per-stage latency percentiles, TTFT stats and
prompt-size distribution so a show-hour can be judged without eyeballing
JSONL. Stdlib only; safe to run on the kiosk box.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[idx]


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("logs/traces.jsonl")
    if not path.exists():
        print(f"no trace file at {path}")
        return 1

    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print(f"trace file {path} has no usable rows")
        return 1

    paths = Counter(r.get("path", "?") for r in rows)
    print(f"file   : {path}  ({len(rows)} turns)")
    print("paths  : " + ", ".join(f"{p}={c}" for p, c in paths.most_common()))
    total = sum(paths.values())
    fast = paths.get("situation", 0) + paths.get("answer-cache", 0)
    print(f"fast-path share: {fast}/{total} = {fast / total:.0%}")

    stages: dict[str, list[float]] = {}
    ttfts: list[float] = []
    prompts: list[float] = []
    for r in rows:
        marks = r.get("stages_ms", {})
        for stage, ms in marks.items():
            stages.setdefault(stage, []).append(float(ms))
        if isinstance(r.get("ttft_s"), (int, float)):
            ttfts.append(float(r["ttft_s"]) * 1000)
        if isinstance(r.get("prompt_chars"), int):
            prompts.append(float(r["prompt_chars"]))

    print("\nstage          n    p50_ms   p95_ms")
    for stage in sorted(stages):
        xs = stages[stage]
        print(
            f"{stage:<14} {len(xs):>4}  {_pctl(xs, 0.50):>7.0f}  {_pctl(xs, 0.95):>7.0f}"
        )
    if ttfts:
        print(
            f"\nllm_ttft       {len(ttfts):>4}  {_pctl(ttfts, 0.50):>7.0f}  {_pctl(ttfts, 0.95):>7.0f}"
        )
    if prompts:
        print(
            f"prompt_chars   {len(prompts):>4}  {_pctl(prompts, 0.50):>7.0f}  {_pctl(prompts, 0.95):>7.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
