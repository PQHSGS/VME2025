"""Idle-attract mode helpers (voice loop invitation logic)."""

import types

from orchestrator import ConversationOrchestrator, _next_attract_line


def make_orch(**over):
    cfg = types.SimpleNamespace(
        attract_enabled=True,
        attract_after_min=5.0,
        attract_lines="xin chào|hai ba",
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    orch = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orch.cfg = cfg
    orch.tts = types.SimpleNamespace(busy=False, disabled=False)
    orch.stt = None
    return orch


def test_rotation_wraps():
    lines = ["a", "b", "c"]
    seq = []
    idx = 0
    for _ in range(5):
        line, idx = _next_attract_line(lines, idx)
        seq.append(line)
    assert seq == ["a", "b", "c", "a", "b"]


def test_rotation_empty_lines():
    assert _next_attract_line([], 0) == ("", 0)


def test_attract_disabled_by_zero_threshold():
    orch = make_orch(attract_after_min=0.0)
    assert not orch._attract_due(last_activity=0.0)


def test_attract_disabled_by_flag():
    orch = make_orch(attract_enabled=False)
    assert not orch._attract_due(last_activity=0.0)


def test_attract_not_due_when_recently_active(monkeypatch):
    orch = make_orch()
    import orchestrator as om

    monkeypatch.setattr(om.time, "monotonic", lambda: 600.0)
    # last activity 1 minute ago vs 5-minute threshold
    assert not orch._attract_due(last_activity=600.0 - 60.0)


def test_attract_due_after_idle_gap(monkeypatch):
    orch = make_orch()
    import orchestrator as om

    now = 1000.0
    monkeypatch.setattr(om.time, "monotonic", lambda: now)
    assert orch._attract_due(last_activity=now - 301.0)


def test_lines_parsed_from_pipe_string():
    orch = make_orch()
    assert orch._attract_lines() == ["xin chào", "hai ba"]
