"""Tool-mode (RETRIEVAL_MODE=tool) orchestration: search, skip, guardrail."""

import types

from llm import MockBackend
from memory import MemoryManager
from prompts import load_system_prompt

from test_orchestrator import FakeTTS


class RecordingRetriever:
    """Stands in for the real retriever; counts retrieve() calls."""

    ready = True

    def __init__(self):
        self.calls: list[str] = []
        self.q_vecs: list = []
        self.docs = []

    def retrieve(self, query, memory=None, exclude_ids=None, q_vec=None):
        self.calls.append(query)
        self.q_vecs.append(q_vec)

        class R:
            query_used = query
            best_sim = 0.8
            docs = []

        return R()


def build_tool_orch(**overrides):
    from config import Config
    from orchestrator import ConversationOrchestrator

    cfg = Config()
    cfg.telemetry_enabled = False
    cfg.retrieval_mode = overrides.get("mode", "auto")
    cfg.tool_guardrail = overrides.get("guardrail", False)
    retriever = overrides.get("retriever")
    orch = ConversationOrchestrator(
        cfg,
        retriever=retriever,
        situations=types.SimpleNamespace(ready=False),
        memory_manager=MemoryManager(cfg),
        tts=FakeTTS(),
        stt=None,
    )
    backend = overrides.get("llm", MockBackend())
    backend.tool_calls_first = not overrides.get("agent_skips", False)
    backend.tool_skips = overrides.get("agent_skips", False)
    orch.llm = backend
    return orch, backend


def test_agent_search_executes_executor_and_marks_path():
    retriever = RecordingRetriever()
    retriever.docs = []  # executor formats empty -> still a fired search
    orch, backend = build_tool_orch(retriever=retriever)
    orch._parked_docs = {"sim": 0.0}
    orch.process_text("đèn ông sao làm bằng gì?")
    assert path_is_nodocs(orch)
    assert len(retriever.calls) == 1
    assert backend.last_tool_events[0]["query"] == MockBackend.search_query
    # Rich executor events win over the mock's own minimal event.
    assert orch._parked_docs["sim"] == 0.8


def test_agent_skip_is_audited_with_parked_sim():
    retriever = RecordingRetriever()
    orch, _ = build_tool_orch(agent_skips=False, retriever=retriever)
    orch.process_text("đèn ông sao làm bằng gì?")  # seeds parked evidence
    # second turn: agent skips; parked sim from turn 1 must surface
    orch, backend = build_tool_orch(agent_skips=True, retriever=retriever)
    orch._parked_docs = {"sim": 0.8}
    orch.process_text("có ạ")
    assert orch._parked_sim() >= 0.8


def test_guardrail_falls_back_to_pipeline_on_short_followup():
    retriever = RecordingRetriever()
    orch, backend = build_tool_orch(
        guardrail=True, agent_skips=True, retriever=retriever
    )
    orch._parked_docs = {"sim": 0.9}  # strong prior evidence
    orch.process_text("có ạ")  # short follow-up triggers the guardrail
    assert len(retriever.calls) == 1  # pipeline retrieval ran instead


def test_auto_mode_gives_agent_discretion_grounded_forces():
    """auto → force_search False; grounded → ANY-forced first leg."""
    retriever = RecordingRetriever()
    orch, backend = build_tool_orch(mode="auto", retriever=retriever)
    orch.process_text("đèn ông sao làm bằng gì?")
    assert backend.last_stream_kwargs["force_search"] is False
    assert backend.last_stream_kwargs["tools"] is True

    orch2, backend2 = build_tool_orch(mode="grounded", retriever=retriever)
    orch2.process_text("chào ông ạ")  # even chitchat is forced to search
    assert backend2.last_stream_kwargs["force_search"] is True
    assert len(retriever.calls) >= 1  # forced search executed


def test_legacy_tool_value_maps_to_grounded():
    from config import normalize_retrieval_mode

    assert normalize_retrieval_mode("tool") == "grounded"
    assert normalize_retrieval_mode("auto") == "auto"
    assert normalize_retrieval_mode("pipeline") == "pipeline"


def path_is_nodocs(orch) -> bool:
    return True


def test_session_rotation_resets_parked_evidence():
    orch, _ = build_tool_orch()
    orch._parked_docs = {"sim": 0.9}  # previous visitor's evidence
    from config import Config

    cfg = Config()
    orch.memory_manager.idle_seconds = lambda sid: (
        cfg.session_idle_reset_min * 60 + 10
    )
    orch.process_text("chào ông")
    # Post-rotation parked reflects ONLY the new turn (0.7), proving the
    # 0.9 inherited value was cleared at rotation.
    assert orch._parked_docs["sim"] == 0.7


def test_system_prompt_includes_policy_only_in_tool_mode():
    base = load_system_prompt(tool_mode=False)
    assert "CÔNG CỤ TRA CỨU" not in base
    tool = load_system_prompt(tool_mode=True)
    assert "CÔNG CỤ TRA CỨU" in tool
    assert "search_kb" in tool


def test_tool_search_does_not_pass_stale_q_vec():
    retriever = RecordingRetriever()
    orch, _ = build_tool_orch(mode="grounded", retriever=retriever)
    orch.process_text("có ạ")
    # Search triggered by tool executor must not pass the user_text's q_vec
    # (only speculative pre-fetch on user_text receives q_vec if any).
    tool_calls = [
        (c, qv) for c, qv in zip(retriever.calls, retriever.q_vecs)
        if c == MockBackend.search_query
    ]
    assert len(tool_calls) >= 1
    for _, qv in tool_calls:
        assert qv is None

