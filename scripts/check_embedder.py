"""One-off sanity check for the new default embedder (SEA-LION-E5-600M).

Verifies: model loads, dimension is sane, and related Vietnamese pairs score
higher than unrelated ones. Run with the .vme_tsg env python.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from rag.embedder import SentenceTransformersEmbedder

cfg = Config()
emb = SentenceTransformersEmbedder(cfg.embed_model, query_prompt=cfg.embed_query_prompt)
print(f"model: {cfg.embed_model} | query prompt: {cfg.embed_query_prompt!r}")

query = "đèn ông sao làm bằng gì?"
docs = [
    "Đèn ông sao làm bằng tre và giấy bóng kính, khung đèn gồm năm cánh sao.",  # related
    "Múa lân là nghệ thuật múa hình thú phổ biến trong lễ hội Tết Trung Thu.",  # topical
    "Bánh dẻo có vỏ làm từ bột nếp rang, nhân đậu xanh và trứng muối.",  # unrelated-ish
]
vecs = emb.encode(docs)
q = emb.encode_query(query)
sims = vecs @ q
for text, s in zip(docs, sims):
    print(f"  {s:.3f}  {text[:50]}")
margin = float(sims[0] - max(sims[1], sims[2]))
assert margin > 0.2, f"gold doc must clearly win (margin={margin:.3f})"
print(f"dim={emb.dim} | OK: gold wins by {margin:+.3f}")
