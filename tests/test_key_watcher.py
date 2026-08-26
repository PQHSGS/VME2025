"""ENTER watcher: one press per physical down-edge; repeats never count."""

from audio import EnterKeyWatcher


def make_watcher(monkeypatch, states, tick_ms=30):
    """Drive _poll_once over a scripted is_down() sequence with a fake clock."""
    import audio as audio_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(audio_mod.time, "monotonic", lambda: clock["t"])

    w = EnterKeyWatcher()
    assert w._available
    seq = iter(states)
    monkeypatch.setattr(w, "is_down", lambda: next(seq))

    accepted = 0
    for _ in states:
        if w._poll_once():
            accepted += 1
        clock["t"] += tick_ms / 1000.0
    return w, accepted


def test_held_key_counts_as_one_press(monkeypatch):
    """Holding ENTER samples True forever after one edge -> exactly 1 press."""
    w, _ = make_watcher(monkeypatch, [True] * 50)  # held ~1.5s
    assert w.consume_press() is True
    assert w.consume_press() is False


def test_two_deliberate_taps_count_twice(monkeypatch):
    """Tap, release past the debounce window, tap again -> 2 presses.

    This is the stop-recording path: the second tap MUST register even
    though the first press was recent.
    """
    states = (
        [True] + [False] * 10  # tap 1 (~0.3s held+released)
        + [False] * 10  # quiet gap > _DEBOUNCE_S
        + [True] + [False] * 10  # tap 2 = "stop and send"
        + [False] * 5
    )
    w, accepted = make_watcher(monkeypatch, states)
    assert accepted == 2
    assert w.consume_press() is True
    assert w.consume_press() is True
    assert w.consume_press() is False


def test_bounce_within_debounce_counts_once(monkeypatch):
    """Contact jitter (quick down-up-down inside 180ms) is ONE press."""
    states = [True, False, True, False] + [False] * 20
    w, _ = make_watcher(monkeypatch, states)
    assert w.consume_press() is True
    assert w.consume_press() is False


def test_drain_clears_backlog(monkeypatch):
    w, _ = make_watcher(monkeypatch, [True])
    assert w.consume_press() is True
    w._presses = 3  # simulate a pile-up from before a fix
    w.drain()
    assert w.consume_press() is False
