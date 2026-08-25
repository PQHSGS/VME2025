"""RAG quality benchmark: hit@k and MRR over a golden QA set.

Usage:
    python scripts/bench_rag.py                 # built-in golden set
    python scripts/bench_rag.py my_qa.csv       # columns: query,expected_path

Requires the FAISS index (python -m rag.ingest) and the embedding model.
Exit code 1 when quality drops below thresholds so CI can gate on it.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN = [
    ("đèn ông sao làm bằng gì vậy ông", ["Đèn ông sao", "trăng"]),
    ("sự tích chú Cuội kể cho cháu nghe", ["Cuội"]),
    ("chị Hằng và chú Cuội có liên quan gì nhau", ["Hằng", "Cuội"]),
    ("bánh nướng khác bánh dẻo thế nào", ["bánh nướng", "bánh dẻo"]),
    ("múa lân sư rông nghĩa là gì", ["lân"]),
    ("tết trung thu có nguồn gốc từ đâu", ["nguồn gốc", "lịch sử"]),
    ("mâm ngũ quả bày trái cây gì", ["ngũ quả"]),
    ("bảo tàng dân tộc học ở đâu", ["Bảo tàng", "bảo tàng"]),
    ("tiến sĩ giấy là ai", ["Tiến sĩ"]),
    ("trò chơi dân gian nào chơi dịp trung thu", ["trò chơi", "ô ăn quan"]),
    ("rối nước là gì vậy ông", ["rối nước"]),
    ("đèn kéo quân diễn như thế nào", ["kéo quân"]),
]


def load_queries(path: str | None):
    if not path:
        return GOLDEN
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        (r["query"], [p.strip() for p in r["expected_path"].split("|")])
        for r in rows
        if r.get("query")
    ]


def main() -> int:
    from config import Config
    from rag.embedder import SentenceTransformersEmbedder
    from rag.retriever import Retriever

    cfg = Config()
    retriever = Retriever(
        cfg,
        SentenceTransformersEmbedder(
            cfg.embed_model,
            query_prompt=cfg.embed_query_prompt,
            num_threads=cfg.embed_threads,
        ),
    )
    if not retriever.load():
        print("!! index missing - run `python -m rag.ingest` first")
        return 2

    queries = load_queries(sys.argv[1] if len(sys.argv) > 1 else None)
    k = cfg.retriever_final_docs
    # Warm-up: model load, first-touch MMR chunk encodes, torch threads -
    # none of these are steady-state per-turn costs.
    retriever.warm_vectors().join()
    retriever.retrieve(queries[0][0])
    hits_at_k, rr_total, slow = 0, 0.0, []
    print(f"{'query':<44} {'hit@%d' % k}  rr     ms")
    for query, expected in queries:
        started = time.perf_counter()
        result = retriever.retrieve(query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        slow.append(elapsed_ms)
        texts = " ".join(d.text.lower() for d in result.docs)
        rank = next(
            (
                i + 1
                for i, d in enumerate(result.docs)
                if any(
                    e.lower() in d.path.lower() or e.lower() in texts for e in expected
                )
            ),
            None,
        )
        hit = rank is not None
        hits_at_k += int(hit)
        rr_total += (1.0 / rank) if hit else 0.0
        flag = "Y" if hit else "-"
        print(
            f"{query[:43]:<44} {flag:^6}  "
            f"{(1.0 / rank) if hit else 0:.2f}   {elapsed_ms:5.0f}"
        )

    n = len(queries)
    hit_rate = hits_at_k / n
    mrr = rr_total / n
    p95 = sorted(slow)[int(0.95 * (len(slow) - 1))]
    print(f"\nhit@{k}={hit_rate:.2f}  mrr={mrr:.2f}  retrieval p95={p95:.0f}ms  n={n}")
    ok = hit_rate >= 0.7 and mrr >= 0.5 and p95 < 400
    print("PASS" if ok else "FAIL (thresholds: hit@k>=0.70 mrr>=0.50 p95<400ms)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
