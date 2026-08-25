"""Time SEA-LION query encoding vs torch thread count (CPU kiosk reality)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import Config
from rag.embedder import SentenceTransformersEmbedder

cfg = Config()
emb = SentenceTransformersEmbedder(cfg.embed_model, query_prompt=cfg.embed_query_prompt)
q = "đèn ông sao làm bằng gì vậy ông?"

print(
    f"torch {torch.__version__} | default threads={torch.get_num_threads()} "
    f"| cpus={__import__('os').cpu_count()}"
)
emb.encode_query(q)  # load + warmup

for n in (1, 2, 4, torch.get_num_threads()):
    torch.set_num_threads(n)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        v = emb.encode_query(q)
        times.append((time.perf_counter() - t0) * 1000)
    print(
        f"threads={n:>2}: {min(times):7.0f} ms  (median-ish {sorted(times)[1]:7.0f} ms)"
    )
