"""Offline tests for the microservice client layer.

All HTTP traffic runs through httpx.MockTransport — no sockets, no models.
"""

import base64
import json

import httpx
import numpy as np
import pytest

from services import clients


# ----------------------------------------------------------------------
# Fake service: one handler emulating every endpoint the clients call.
# ----------------------------------------------------------------------
def make_handler(*, fail=False, docs=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(503, json={"detail": "down"})
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "engine": "fake"})
        if path == "/embed":
            vec = [0.1] * 4
            return httpx.Response(200, json={"vector": vec, "dim": 4})
        if path == "/retrieve":
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "docs": docs or [],
                    "query_used": body["query"],
                    "elapsed_ms": 3,
                    "memory_ctx_seen": (body.get("memory_ctx") or {}).get(
                        "seen_chunk_ids"
                    ),
                },
            )
        if path == "/situation":
            return httpx.Response(
                200,
                json={
                    "matched": True,
                    "question": "chào ông",
                    "answer": "Chào cháu!",
                    "score": 0.99,
                },
            )
        if path == "/situations":
            return httpx.Response(
                200, json={"rows": [{"question": "q", "answer": "a"}], "enabled": True}
            )
        if path == "/transcribe":
            body = json.loads(request.content)
            pcm = np.frombuffer(base64.b64decode(body["audio_b64"]), dtype=np.float32)
            return httpx.Response(200, json={"text": f"got{pcm.size}", "elapsed_ms": 1})
        if path == "/stream":
            # Chunk containing a newline would break naive SSE framing.
            chunks = ["Vâng,", " dòng có xuống hàng\nthứ hai", " xong."]
            lines = [
                f"data: {json.dumps({'t': c}, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
                for c in chunks
            ] + [b'data: {"done": true}\n\n']

            def gen():
                yield from lines

            return httpx.Response(
                200,
                content=gen(),
                headers={"content-type": "text/event-stream"},
            )
        if path == "/complete":
            return httpx.Response(200, json={"text": "hoàn tất", "elapsed_ms": 2})
        if path == "/synthesize":
            pcm = np.array([100, -100, 300], dtype=np.int16)
            return httpx.Response(
                200,
                json={
                    "audio_b64": base64.b64encode(pcm.tobytes()).decode(),
                    "sample_rate": 24000,
                    "engine": "fake",
                    "elapsed_ms": 1,
                    "cached": False,
                },
            )
        return httpx.Response(404)

    return handler


def remote_cfg(monkeypatch):
    cfg = type("Cfg", (), {})()
    cfg.rag_service_url = "http://rag.test"
    cfg.asr_service_url = "http://asr.test"
    cfg.llm_service_url = "http://llm.test"
    cfg.tts_service_url = "http://tts.test"
    cfg.dedup_window_turns = 3
    cfg.situations_enabled = True
    return cfg


@pytest.fixture()
def transport():
    return httpx.MockTransport(make_handler())


# ----------------------------------------------------------------------
def test_remote_llm_stream_parses_json_sse_with_newlines(transport):
    llm = clients.RemoteLLM(remote_cfg(None), transport=transport)
    out = list(llm.stream([{"role": "user", "content": "hi"}]))
    assert out == ["Vâng,", " dòng có xuống hàng\nthứ hai", " xong."]


def test_remote_llm_complete(transport):
    llm = clients.RemoteLLM(remote_cfg(None), transport=transport)
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hoàn tất"


def test_remote_llm_raises_when_service_down():
    bad = httpx.MockTransport(make_handler(fail=True))
    llm = clients.RemoteLLM(remote_cfg(None), transport=bad)
    with pytest.raises(RuntimeError):
        list(llm.stream([{"role": "user", "content": "hi"}]))


def test_remote_embedder_returns_vector_and_dim(transport):
    emb = clients.RemoteEmbedder(remote_cfg(None), transport=transport)
    vec = emb.encode_query("xin chào")
    assert vec.shape == (4,)
    assert emb.dim == 4


def test_remote_retriever_maps_docs_and_memory_ctx():
    docs = [
        {
            "chunk_id": "c1",
            "path": "kb.txt",
            "text": "đèn ông sao",
            "score": 0.9,
        }
    ]
    retriever = clients.RemoteRetriever(
        remote_cfg(None), transport=httpx.MockTransport(make_handler(docs=docs))
    )

    class Mem:
        def looks_like_followup(self, q):
            return True

        def last_topics(self, n=2):
            return "trung thu"

        def recently_seen_chunk_ids(self, w):
            return {"c9"}

    result = retriever.retrieve("nó ở đâu?", memory=Mem())
    assert result.query_used == "nó ở đâu?"
    assert result.docs[0].chunk_id == "c1"


def test_remote_retriever_degrades_to_empty_on_failure():
    bad = httpx.MockTransport(make_handler(fail=True))
    retriever = clients.RemoteRetriever(remote_cfg(None), transport=bad)
    result = retriever.retrieve("q")
    assert result.docs == []


def test_remote_situations_match(transport):
    sit = clients.RemoteSituations(remote_cfg(None), transport=transport)
    assert sit.load() is True
    hit = sit.match("chào ông")
    assert hit is not None and hit.answer == "Chào cháu!" and hit.score > 0.9


def test_remote_stt_roundtrip(transport):
    stt = clients.RemoteSTT(remote_cfg(None), transport=transport)
    audio = np.zeros(160, dtype=np.float32) + 0.5
    text = stt.transcribe(audio, 16000)
    assert text == "got160"
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""


def test_remote_synth_decodes_pcm(transport):
    synth = clients.RemoteSynth(remote_cfg(None), transport=transport)
    pcm, rate = synth("xin chào")
    assert rate == 24000
    assert pcm.dtype == np.int16 and pcm[0] == 100


def test_remote_synth_raises_on_error_payload():
    handler = make_handler()

    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/synthesize":
            return httpx.Response(200, json={"error": "engine exploded"})
        return handler(request)

    synth = clients.RemoteSynth(
        remote_cfg(None), transport=httpx.MockTransport(failing)
    )
    with pytest.raises(RuntimeError, match="exploded"):
        synth("text")


def test_build_remote_tts_player_uses_local_playback():
    """The kiosk keeps TTSPlayer (queue/barge-in); only synthesis is remote."""
    from tts import TTSPlayer

    cfg = remote_cfg(None)
    cfg.tts_enabled = True
    cfg.vienneu_voice = "X"
    cfg.vienneu_backend = "onnx"
    player = clients.build_remote_tts_player(
        cfg, transport=httpx.MockTransport(make_handler())
    )
    assert isinstance(player, TTSPlayer)
    assert player._synth_pcm is not None
