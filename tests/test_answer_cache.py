"""Offline tests for the semantic answer cache + TTFA first-clause splitter."""

import types

import numpy as np

from answer_cache import AnswerCache, is_cacheable_query, normalize
from sentences import SentenceSplitter


def make_cfg(**over):
    base = dict(
        answer_cache_enabled=True,
        answer_cache_similarity=0.92,
        answer_cache_max_entries=3,
        answer_cache_ttl_min=240,
        answer_cache_min_reply_chars=30,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class OverlapEmbedder:
    """Deterministic hashed bag-of-words; cosine ~ token overlap."""

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        self.calls += 1
        v = np.zeros(64, dtype=np.float32)
        for w in normalize(text).split():
            idx = int.from_bytes(w.encode("utf-8")[:4], "little") % 64
            v[idx] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_query(self, text):
        return self._vec(text)


def test_normalize_strips_punct_and_case():
    assert normalize("Đèn Ông Sao? ") == normalize("đèn ông sao")


def test_followup_queries_never_cacheable():
    assert not is_cacheable_query("còn trẻ em thì sao?")
    assert not is_cacheable_query("vậy giá vé")
    assert is_cacheable_query("đèn ông sao làm bằng gì vậy?")


def test_exact_hit_skips_embedder():
    emb = OverlapEmbedder()
    cache = AnswerCache(make_cfg(), emb)
    reply = "Đèn ông sao làm bằng tre và giấy bóng kính, khung năm cánh."
    cache.store("Đèn ông sao làm bằng gì?", reply)
    calls_before = emb.calls
    assert cache.lookup("ĐÈN ÔNG SAO LÀM BẰNG GÌ") == reply
    # exact tier must not pay for an embedding
    assert emb.calls == calls_before  # encode happens in store only


def test_semantic_hit_above_threshold():
    cfg = make_cfg(answer_cache_similarity=0.6)
    cache = AnswerCache(cfg, OverlapEmbedder())
    reply = "Múa lân diễn ra trong đêm hội Trung Thu tại sân bảo tàng, miễn phí."
    cache.store("múa lân diễn ở đâu giờ nào?", reply)
    hit = cache.lookup("múa lân diễn ở đâu và giờ nào?")
    assert hit == reply


def test_no_semantic_hit_below_threshold():
    cfg = make_cfg(answer_cache_similarity=0.99)
    cache = AnswerCache(cfg, OverlapEmbedder())
    cache.store(
        "giá vé vào cửa bao nhiêu?", "Vé vào cửa hoàn toàn miễn phí cho mọi khách."
    )
    assert cache.lookup("bảo tàng mở cửa đến mấy giờ?") is None


def test_lru_eviction_bounds_entries():
    cache = AnswerCache(make_cfg(), OverlapEmbedder())
    for i in range(5):
        cache.store(
            f"câu hỏi số {i} về bảo tàng?", f"trả lời {i} với nội dung dài đủ chuẩn."
        )
    assert len(cache) <= 3


def test_short_or_truncated_replies_rejected():
    cache = AnswerCache(make_cfg(), OverlapEmbedder())
    cache.store("múa rối nước là gì vậy nhỉ?", "ngắn quá")  # below min chars
    assert len(cache) == 0
    assert cache.lookup("múa rối nước là gì vậy nhỉ?") is None


# ---------------------------------------------------------------------------
# SentenceSplitter: first-clause early emission (TTFA)
# ---------------------------------------------------------------------------
def test_first_clause_emitted_at_first_comma():
    sp = SentenceSplitter(early_first_clause=True)
    out = sp.push("Đúng, đèn ông sao được làm từ tre")
    assert out == ["Đúng"]
    rest = sp.push(" và giấy bóng kính. Bên dưới gắn chuông nhỏ.") + sp.flush()
    assert any("đèn ông sao" in s for s in rest)


def test_first_clause_disabled_by_default():
    sp = SentenceSplitter()
    out = sp.push("Dạ, ông nghe đây nhé. Câu này kết thúc đủ.")
    assert out[0].startswith("Dạ, ông nghe đây nhé")


def test_first_clause_only_once():
    sp = SentenceSplitter(early_first_clause=True)
    first = sp.push("Đúng, ông nói tiếp đây")
    assert first == ["Đúng"]
    # after the clause was emitted, later commas must NOT split again
    second = sp.push(" với nhiều chi tiết, và câu kết thúc ở đây.")
    assert all(", " not in s or "chi tiết, và" in s for s in second + sp.flush())


def test_clause_window_bounded():
    sp = SentenceSplitter(early_first_clause=True)
    long_opening = "x" * 90 + ", rồi mới có dấu phẩy sau đó"
    out = sp.push(long_opening)
    assert out == []  # comma outside the 80-char window -> no early emit


def test_chunk_ids_roundtrip_for_dedup_remark():
    import numpy as np

    class Emb:
        def encode_query(self, text):
            v = np.ones(8, dtype=np.float32)
            return v / np.linalg.norm(v)

    cfg = types.SimpleNamespace(
        answer_cache_enabled=True,
        answer_cache_similarity=0.9,
        answer_cache_max_entries=10,
        answer_cache_ttl_min=60,
        answer_cache_min_reply_chars=5,
    )
    cache = AnswerCache(cfg, Emb())
    cache.store(
        chr(273) + 'en ' + chr(244) + 'ng sao l' + chr(224) + 'm b' + chr(7857) + 'ng g' + chr(236) + '?',
        'B' + chr(7857) + 'ng tre v' + chr(224) + ' gi' + chr(7855) + 'y, nh' + chr(7865) + ' m' + chr(224) + ' p' + chr(7867) + 'p.',
        chunk_ids=['c1', 'c2'],
    )
    hit = cache.lookup(chr(273) + 'EN ONG SAO LAM BANG GI')
    assert hit
    assert cache.last_hit_chunk_ids == ['c1', 'c2']
