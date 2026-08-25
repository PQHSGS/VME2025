from rag.embedder import FakeEmbedder
from rag.situations import SituationMatcher, _normalize


class MiniCfg:
    situations_csv = None
    situations_enabled = True
    situations_threshold = 0.86


ROWS = [
    {"question": "Con chào ông ạ", "guidance": "", "answer": "Chào cháu nhé!"},
    {"question": "Ông là ai ạ?", "guidance": "", "answer": "Ông là Tiến sĩ Giấy."},
]


def build_matcher():
    m = SituationMatcher(MiniCfg(), FakeEmbedder())
    m.rows = ROWS
    m._exact = {_normalize(r["question"]): r for r in ROWS}
    m.vectors = m.embedder.encode([r["question"] for r in ROWS], normalize=True)
    return m


def test_exact_match_bypasses_embeddings():
    m = build_matcher()
    hit = m.match("Con chào ông ạ")
    assert hit and hit.answer == "Chào cháu nhé!" and hit.score == 1.0


def test_exact_match_normalizes_punctuation_case():
    m = build_matcher()
    assert m.match("CON CHÀO ÔNG Ạ !") is not None
    assert m.match("  con chào ông ạ. ") is not None


def test_unrelated_query_misses():
    m = build_matcher()
    assert m.match("bánh chưng gói lá chuối") is None


def test_disabled_flag_short_circuits():
    class Off(MiniCfg):
        situations_enabled = False

    m = SituationMatcher(Off(), FakeEmbedder())
    assert not m.ready and m.match("anything") is None
