import numpy as np

from rag.embedder import FakeEmbedder
from rag.retriever import Retriever


class MiniCfg:
    domain_keywords = ["trung thu", "đèn ông sao"]
    gate_threshold = 0.40
    retriever_topk_candidates = 4
    retriever_final_docs = 2
    retriever_min_score = 0.05
    retriever_mmr_lambda = 1.0  # pure relevance -> deterministic MMR
    context_char_budget = 10_000
    dedup_penalty = 0.5
    dedup_window_turns = 3


class FakeIndex:
    """IndexFlatIP lookalike over precomputed unit vectors."""

    def __init__(self, vectors: np.ndarray):
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ntotal = int(vectors.shape[0])

    def search(self, query, k):
        scores = self._vectors @ query[0]
        order = np.argsort(-scores)[:k]
        return scores[order][None, :], order[None, :]


def build_retriever(texts, cfg=MiniCfg()):
    emb = FakeEmbedder(dim=64)
    r = Retriever(cfg, emb)
    vecs = emb.encode([f"[p] {t}" for t in texts], normalize=True)
    r.index = FakeIndex(vecs)
    r.docs_meta = [
        {"chunk_id": str(i), "path": f"P{i}", "title": f"T{i}", "text": t}
        for i, t in enumerate(texts)
    ]
    r.centroids = None
    return r, emb


DOCS = [
    "Đèn ông sao có năm cánh sao lớn.",
    "Đèn ông sao được làm bằng giấy và tre.",
    "Đèn ông sao gắn với đêm hội trăng rằm.",
]


def test_keyword_gate_hits():
    r, _ = build_retriever(DOCS)
    assert r.should_retrieve("đèn ông sao làm từ gì vậy")
    assert not r.should_retrieve("chào ông")


def test_retrieve_ranks_by_similarity():
    r, _ = build_retriever(DOCS)
    res = r.retrieve("đèn ông sao năm cánh sao", force=True)
    assert res.docs and res.docs[0].text.startswith("Đèn ông sao")


def test_dedup_penalty_demotes_recent_chunk():
    class PenalCfg(MiniCfg):
        dedup_penalty = 5.0  # large: seen chunk falls below min_score

    r, _ = build_retriever(DOCS, PenalCfg())

    class Mem:
        turn_count = 2

        def recently_seen_chunk_ids(self, w):
            return {"0"}

    res = r.retrieve("đèn ông sao năm cánh sao", force=True, memory=Mem())
    top = [d.chunk_id for d in res.docs]
    assert top and "0" not in top[:1], "recently shown chunk should be demoted"


def test_char_budget_trims_docs():
    class TinyCfg(MiniCfg):
        context_char_budget = 30

    long_docs = ["câu rất dài " * 20 for _ in range(3)]
    r, _ = build_retriever(long_docs, TinyCfg())
    res = r.retrieve("câu rất dài", force=True)
    assert len(res.docs) == 1  # trimmed to the single best


def test_min_score_filters_garbage():
    r, _ = build_retriever(DOCS)
    res = r.retrieve("mua bánh mì ở đâu", force=True)  # unrelated
    assert all(d.score >= MiniCfg.retriever_min_score for d in res.docs)


def test_not_ready_returns_empty():
    emb = FakeEmbedder()
    r = Retriever(MiniCfg(), emb)
    assert not r.ready
    res = r.retrieve("anything", force=True)
    assert res.docs == []
