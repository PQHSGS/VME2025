import numpy as np

from tts import TARGET_RATE, TTSPlayer


class MiniCfg:
    tts_enabled = True
    tts_voice = "vi-VN-NamMinhNeural"
    tts_rate = "+10%"
    tts_cache_size = 8
    vienneu_voice = "Minh Đức"
    vienneu_backend = "onnx"
    tts_synth_workers = 2
    tts_max_consecutive_failures = 3


def make_player():
    p = TTSPlayer(MiniCfg())
    assert not p.disabled
    return p


def test_submit_tracks_submitted_with_tags():
    p = make_player()
    assert p.submit("Câu một.", tag="filler")
    assert p.submit("Câu hai.", tag="reply")
    # drain what workers would have consumed (player not started)
    items = []
    while not p._text_queue.empty():
        items.append(p._text_queue.get_nowait())
    assert len(p._submitted) == 2
    assert [tag for tag, _ in p._submitted] == ["filler", "reply"]


def test_heard_text_reflects_playback_started_only():
    p = make_player()
    p.submit("Một.", "reply")
    p.submit("Hai.", "reply")
    p.submit("Ba.", "reply")
    # simulate play_worker marking the first sentence as started
    with p._book_lock:
        if p._submitted:
            p._heard.append(p._submitted.popleft())
    assert p.heard_text(tag="reply") == "Một."
    p.stop()
    # stop() dropped the remaining unheard sentences from bookkeeping
    assert len(p._submitted) == 0
    assert p.heard_text() == "Một."
    assert not p.speaking


def test_reset_reply_bookkeeping_clears_all():
    p = make_player()
    p.submit("X.", "reply")
    p.submit("Y.", "filler")
    p.reset_reply_bookkeeping()
    assert p.heard_text() == "" and len(p._submitted) == 0


def test_disabled_player_rejects():
    class Off(MiniCfg):
        tts_enabled = False

    p = TTSPlayer(Off())
    assert p.disabled
    assert p.submit("không ai nghe") is False


def test_pcm_synth_path_and_prewarm_cache():
    calls: list[str] = []

    def fake_pcm(text: str):
        calls.append(text)
        return np.ones(480, dtype=np.int16), 24000

    p = TTSPlayer(MiniCfg(), synth_pcm_fn=fake_pcm)
    assert "vienneu" in p.engine_name
    # prewarm synthesizes only non-empty lines and skips playback queues
    assert p.prewarm(["Chào cháu.", "", "   "]) == 1
    assert not p.busy
    assert any(k.startswith("vienneu|") for k in p._cache)
    # second synthesis of the same text hits the cache (no new call)
    p.prewarm(["Chào cháu."])
    assert calls.count("Chào cháu.") == 1


def test_resample_identity_and_change():
    pcm = np.zeros(240, dtype=np.int16)
    assert resample_same(pcm) is pcm
    out = resample_to_16k(pcm)
    assert out.shape[0] == int(240 / TARGET_RATE * 16000)


def resample_same(pcm):
    from tts import resample_to

    return resample_to(pcm, TARGET_RATE, TARGET_RATE)


def resample_to_16k(pcm):
    from tts import resample_to

    return resample_to(pcm, TARGET_RATE, 16000)


def test_out_of_order_synth_releases_in_sequence():
    """Parallel workers finish out of order; playback order is preserved."""
    p = make_player()
    a = np.zeros(10, np.int16)
    b = np.ones(10, np.int16)
    p._file_result(1, 0, (a, TARGET_RATE))  # sentence 2 done first
    assert p._pcm_queue.empty()  # held back: sentence 1 still synthesizing
    p._file_result(0, 0, (b, TARGET_RATE))
    first = p._pcm_queue.get_nowait()
    second = p._pcm_queue.get_nowait()
    assert first[0] is b and second[0] is a
    p._file_result(2, 0, None)  # failed slot must not block the queue
    assert p._pcm_queue.get_nowait() is None


def test_stale_generation_results_discarded():
    """Results in flight during a barge-in never reach the new reply."""
    p = make_player()
    p._gen = 3
    p.stop()
    assert p._gen == 4
    a = np.zeros(10, np.int16)
    p._file_result(0, 3, (a, TARGET_RATE))  # old generation
    assert p._pcm_queue.empty()
    p._file_result(0, 4, (a, TARGET_RATE))  # current generation passes
    assert p._pcm_queue.get_nowait()[0] is a
