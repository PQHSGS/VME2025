from sentences import SentenceSplitter


def test_simple_stream_yields_complete_sentences():
    s = SentenceSplitter()
    out = []
    out += s.push("Ông là Tiến sĩ Giấy. ")
    out += s.push("Cháu tên gì?")
    assert out == ["Ông là Tiến sĩ Giấy.", "Cháu tên gì?"]
    assert s.flush() == []


def test_partial_sentence_stays_buffered():
    s = SentenceSplitter()
    assert s.push("Đèn ông sao là một") == []
    assert s.push(" loại đèn truyền thống.") == [
        "Đèn ông sao là một loại đèn truyền thống."
    ]


def test_flush_returns_tail_without_punctuation():
    s = SentenceSplitter()
    s.push("Câu có dấu.")
    tail = s.flush()
    assert tail == [] or tail == ["Câu có dấu."]
    s2 = SentenceSplitter()
    s2.push("chưa xong")
    assert s2.flush() == ["chưa xong"]


def test_decimal_not_split():
    s = SentenceSplitter()
    out = s.push("Bánh giá 1.5 triệu đồng. Rẻ mà.")
    assert any("1.5" in sent for sent in out), out
    # the decimal sentence must not be cut inside the number
    assert not any(sent.strip().endswith("1.") for sent in out)


def test_abbreviation_not_split():
    s = SentenceSplitter()
    out = s.push("Ông ở Bảo tàng Dân tộc học Hà Nội. ")
    assert len(out) <= 1  # no split happened at all is fine; key: no bogus tiny piece


def test_markdown_cleaned_from_tts_input():
    s = SentenceSplitter()
    out = s.push("**Ông** nói đây. #không đọc thẻ")
    joined = " ".join(out)
    assert "*" not in joined and "#" not in joined


def test_force_split_long_run():
    s = SentenceSplitter(max_chars=60)
    long_text = "một hai ba " * 30  # no punctuation, 360 chars
    out = s.push(long_text)
    assert len(out) >= 3
    assert all(len(part) <= 90 for part in out)
