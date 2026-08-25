"""Debounced ENTER watcher: holding the key must count as ONE press."""

from audio import EnterKeyWatcher


class FakeMsvcrt:
    """Scripted keys, no keyboard needed."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.i = 0

    def kbhit(self):
        return self.i < len(self.keys)

    def getwch(self):
        key = self.keys[self.i]
        self.i += 1
        return key


def make_watcher(monkeypatch, keys, tick_ms=30, key_down=False):
    """Drive _poll_once over a scripted key stream with a fake clock."""
    import audio as audio_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(audio_mod.time, "monotonic", lambda: clock["t"])

    w = EnterKeyWatcher()
    assert w._available
    monkeypatch.setattr(w, "is_down", lambda: key_down)
    kb = FakeMsvcrt(keys)
    accepted = 0
    while kb.kbhit():
        if w._poll_once(kb):
            accepted += 1
        clock["t"] += tick_ms / 1000.0
    return w, accepted


def test_key_repeat_counts_as_one_press(monkeypatch):
    """Holding ENTER streams ~30 \\r per second; debounce+key-state keeps 1."""
    w, _ = make_watcher(monkeypatch, ["\r"] * 50, key_down=True)  # held down
    assert w.consume_press() is True  # exactly one
    assert w.consume_press() is False  # repeat backlog suppressed


def test_two_deliberate_taps_count_twice(monkeypatch):
    # 40 non-ENTER keys between taps at 30ms each > 250ms debounce window.
    keys = ["\r"] + ["x"] * 40 + ["\r"] + ["x"] * 40
    w, _ = make_watcher(monkeypatch, keys)
    assert w.consume_press() is True
    assert w.consume_press() is True
    assert w.consume_press() is False


def test_non_enter_keys_ignored(monkeypatch):
    w, _ = make_watcher(monkeypatch, ["a", "b", " "] * 10)
    assert w.consume_press() is False


def test_drain_clears_backlog(monkeypatch):
    w, _ = make_watcher(monkeypatch, ["\r"])
    assert w.consume_press() is True
    w._presses = 3  # simulate a pile-up from before a fix
    w.drain()
    assert w.consume_press() is False
