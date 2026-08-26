from rag.situations import Situation
from llm import MockBackend
from memory import MemoryManager


class FakeTTS:
    def __init__(self):
        self.submitted = []
        self.disabled = False

    def submit(self, sentence, tag="reply"):
        self.submitted.append((tag, sentence))
        return True

    @property
    def busy(self):
        return False

    speaking = False

    def wait_done(self, timeout=30.0):
        return True

    def reset_reply_bookkeeping(self):
        pass

    def heard_text(self, tag=None):
        return ""

    def start(self):
        pass

    def close(self):
        pass

    def stop(self):
        pass

    def prewarm(self, phrases):
        pass


class FakeSituations:
    ready = True
    rows = [object()]

    def __init__(self, answer="Chào cháu nhé, cháu tên gì?"):
        self.answer = answer

    def match(self, query, q_vec=None):
        return Situation(score=1.0, question=query, guidance="", answer=self.answer)


class ExplodingBackend:
    name = "explode"

    def stream(self, *a, **k):
        raise RuntimeError("api down")

    def complete(self, *a, **k):
        raise RuntimeError("api down")

    def health_check(self):
        return False


def build(**overrides):
    from config import Config
    from orchestrator import ConversationOrchestrator

    cfg = Config()
    cfg.telemetry_enabled = False  # keep test runs quiet on disk
    tts = FakeTTS()
    orch = ConversationOrchestrator(
        cfg,
        retriever=None,
        situations=overrides.get("situations"),
        memory_manager=MemoryManager(cfg),
        tts=tts,
        stt=None,
    )
    orch.llm = overrides.get("llm", MockBackend())
    return orch, tts


def test_situation_fastpath_queues_speech():
    orch, tts = build(situations=FakeSituations())
    reply = orch.process_text("Con chào ông ạ")
    assert reply.startswith("Chào cháu")
    # regression: fast-path answers must actually reach the speaker queue
    assert any(text == reply for _, text in tts.submitted)


def test_llm_path_streams_into_memory():
    orch, tts = build()
    reply = orch.process_text("đèn ông sao làm bằng gì?")
    assert "Trung Thu" in reply or "múa lân" in reply  # canned mock text
    last = list(orch.memory_manager.get(orch.session_id).recent)[-1]
    assert last.bot == reply and last.user.startswith("đèn ông sao")


def test_fallback_on_backend_failure_is_spoken():
    orch, tts = build(llm=ExplodingBackend())
    reply = orch.process_text("kể chuyện đi ông")
    assert reply == orch.cfg.fallback_reply
    assert any(reply == text for _, text in tts.submitted)


def test_circuit_breaker_opens_after_repeated_failures():
    orch, _ = build(llm=ExplodingBackend())
    for _ in range(3):  # threshold = 3 consecutive
        reply = orch.process_text("câu hỏi bất kỳ")
        assert reply == orch.cfg.fallback_reply
    # 4th turn must short-circuit WITHOUT touching the backend again
    calls = {"n": 0}

    class Counting(ExplodingBackend):
        def stream(self, *a, **k):
            calls["n"] += 1
            raise RuntimeError("still down")

    orch.llm = Counting()
    reply = orch.process_text("lần thứ tư")
    assert reply == orch.cfg.fallback_reply
    assert calls["n"] == 0


def test_success_resets_failure_counter():
    orch, _ = build()
    orch.llm_failures.record_failure()  # one failure...
    reply = orch.process_text("đèn ông sao?")  # ...then a clean success
    assert "Trung Thu" in reply or "múa lân" in reply
    assert orch.llm_failures.count == 0


def test_idle_session_reset_drops_previous_facts():
    import time as _time

    orch, _ = build(situations=None)
    sid_first = orch.session_id
    mem = orch.memory_manager.get(sid_first)
    mem.add_user("Cháu tên là Minh ạ")
    mem.last_used = _time.time() - (orch.cfg.session_idle_reset_min * 60 + 30)
    orch.process_text("chào ông")
    assert orch.session_id != sid_first
    fresh = orch.memory_manager.get(orch.session_id)
    assert fresh is not mem and "tên" not in fresh.facts


def test_queue_speech_splits_sentences():
    orch, tts = build()
    orch._queue_speech("Câu một đây. Và câu hai nữa!")
    texts = [t for _, t in tts.submitted]
    assert texts == ["Câu một đây.", "Và câu hai nữa!"]


def test_no_tts_never_crashes():
    orch, _ = build()
    orch.tts = None
    reply = orch.process_text("chào ông")
    assert isinstance(reply, str) and reply


def test_summarize_thread_joined_on_shutdown():
    orch, _ = build()
    mem = orch.memory_manager.get(orch.session_id)
    for i in range(10):
        mem.add_user(f"u{i}")
        mem.add_bot_reply(f"b{i}")
    thread = orch._maybe_summarize(mem)
    if thread is not None:  # gate may skip; either is valid
        orch.join_background_work(timeout=2)


def test_llm_ttft_timeout_triggers_fallback():
    import time

    class StallingBackend:
        name = "stalling"

        def stream(self, *args, **kwargs):
            time.sleep(1.0)
            yield "Quá muộn rồi"

    orch, tts = build(llm=StallingBackend())
    orch.cfg.llm_hard_deadline_s = 0.2  # force tight timeout
    reply = orch.process_text("hỏi một câu")
    assert reply == orch.cfg.fallback_reply
    assert orch.llm_failures.count == 1
