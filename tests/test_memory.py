import time

from memory import Exchange, MemoryManager, SessionMemory


class MiniCfg:
    recent_exchanges = 4
    summarize_every_turns = 3
    summary_max_chars = 700
    session_ttl_minutes = 90


def test_name_and_like_extraction():
    m = SessionMemory("s1", recent_exchanges=4)
    m.add_user("Cháu tên là Minh ạ")
    m.add_user("Cháu thích đèn ông sao lắm ông ơi")
    assert m.facts.get("tên") == "Minh"
    assert "đèn ông sao" in m.facts.get("thích", "")


def test_recent_window_and_amend():
    m = SessionMemory("s2", recent_exchanges=2)
    m.add_user("hỏi một")
    m.add_bot_reply("trả lời một")
    m.add_user("hỏi hai")
    m.add_bot_reply("trả lời hai")
    m.add_user("hỏi ba")
    m.add_bot_reply("trả lời ba")
    assert len(m.recent) == 2  # window enforced
    assert [e.turn for e in m.recent] == [2, 3]
    assert m.amend_last_bot_reply("trả lời ba (bị chen ngang)")
    assert m.recent[-1].bot.endswith("(bị chen ngang)")
    assert not SessionMemory("empty", 2).amend_last_bot_reply("x")


def test_summary_apply_prunes_overflow():
    m = SessionMemory("s3", recent_exchanges=4)
    for i in range(4):
        m.add_user(f"câu {i}")
        m.add_bot_reply(f"đáp {i}")
    overflow = m.overflow_exchanges(keep=1)
    assert len(overflow) == 3 and overflow[0].user == "câu 0"
    m.apply_summary("trẻ hỏi về đèn ông sao", summarized_up_to_turn=overflow[-1].turn)
    assert m.summary == "trẻ hỏi về đèn ông sao"
    assert len(m.recent) == 1 and m.recent[0].turn == 4
    assert m.turns_since_summary == 0


def test_summary_replaces_not_appends_and_respects_cap():
    # The summarizer prompt already merges the old summary into its result,
    # so apply_summary must REPLACE - appending would duplicate context.
    m = SessionMemory("s3b", recent_exchanges=4, summary_max_chars=50)
    m.apply_summary(
        "bản tóm tắt cũ khá dài để kiểm tra giới hạn ký số", summarized_up_to_turn=1
    )
    m.apply_summary("tóm tắt mới", summarized_up_to_turn=2)
    assert m.summary == "tóm tắt mới"
    m.apply_summary("x" * 80, summarized_up_to_turn=3)
    assert len(m.summary) == 50


def test_needs_summary_gate():
    m = SessionMemory("s4", recent_exchanges=4)
    assert not m.needs_summary(summarize_every_turns=6)
    for i in range(7):
        m.add_user(f"u{i}")
        m.add_bot_reply(f"b{i}")
    assert m.needs_summary(summarize_every_turns=6)


def test_seen_chunks_window():
    m = SessionMemory("s5", recent_exchanges=4)
    m.add_user("q1")
    m.mark_chunks_shown(["a", "b"])
    m.add_user("q2")
    m.mark_chunks_shown(["c"])
    m.add_user("q3")
    m.mark_chunks_shown(["d"])
    # window of 1 turn: only the newest chunk is 'recent'
    assert m.recently_seen_chunk_ids(window_turns=1) == {"d"}
    assert {"c", "d"} <= m.recently_seen_chunk_ids(window_turns=2)


def test_followup_heuristics():
    m = SessionMemory("s6", recent_exchanges=2)
    m.add_user("kể về chú Cuội")
    m.add_bot_reply("chú Cuội là...")
    assert m.looks_like_followup("nó ở đâu vậy?")
    assert not m.looks_like_followup("Kể cho cháu nghe toàn bộ sự tích chú Cuội với")
    assert "chú Cuội" in m.last_topics()


def test_manager_ttl_and_cleanup():
    mgr = MemoryManager(MiniCfg())
    mem = mgr.get("kiosk-1")
    mem.last_used = time.time() - 999 * 60
    assert mgr.get("kiosk-1").session_id == "kiosk-1"  # recreated (stale)
    assert mgr.cleanup() >= 0


def test_exchange_defaults():
    e = Exchange(user="u", bot="b", turn=1)
    assert e.at > 0
