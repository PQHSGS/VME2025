from prompts import build_context_block, build_messages


class FakeMemory:
    summary = "trẻ tên Bông, đã hỏi về múa lân"
    facts = {"tên": "Bông", "thích": "múa lân"}

    class _E:
        def __init__(self, user, bot):
            self.user, self.bot = user, bot

    recent = [_E("múa lân là gì ạ", "Múa lân là..."), _E("ai múa", "Các nghệ nhân...")]


DOCS = [
    {
        "path": "Trung Thu > Đèn ông sao",
        "text": "Đèn ông sao làm bằng giấy tre.",
        "score": 0.9,
    }
]


def test_context_block_layers():
    block = build_context_block(FakeMemory(), DOCS)
    assert "TÓM TẮT" in block and "THÔNG TIN ĐÃ BIẾT" in block
    assert "TÀI LIỆU THAM KHẢO" in block and "[1]" in block
    assert "Đèn ông sao làm bằng giấy tre." in block


def test_context_block_empty_when_nothing():
    class Empty:
        summary = ""
        facts = {}
        recent = []

    assert build_context_block(Empty(), None) is None


def test_build_messages_shape_and_history_limit():
    messages, meta = build_messages(
        "SYSTEM", FakeMemory(), "đèn kéo quân là gì?", docs=DOCS, history_limit=1
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant"]
    body = messages[1]["content"]
    # history limited to the LAST exchange: older one dropped, newer kept
    assert "Múa lân là..." not in body and "múa lân là gì" not in body
    assert "ai múa" in body and "Các nghệ nhân..." in body
    assert "HỘI THOẠI GẦN ĐÂY" in body and "CÂU HỎI HIỆN TẠI" in body
    assert "Đèn ông sao làm bằng giấy tre." in body
    # pre-ack keeps model in role; system prompt verbatim
    assert messages[2]["content"].startswith("Ông nghe rồi")
    assert messages[0]["content"] == "SYSTEM"
    assert meta["docs"] == 1 and meta["recent_turns"] == 1
