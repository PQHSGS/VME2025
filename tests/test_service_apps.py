"""Offline tests for the service apps' endpoint wiring.

TestClient is used WITHOUT a context manager so lifespan/startup handlers
(heavy model loads) never run; fakes are injected as module globals.
"""

import base64

import numpy as np
import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


# ----------------------------------------------------------------------
# RAG service: /retrieve must forward memory context to the retriever.
# ----------------------------------------------------------------------
def test_rag_retrieve_forwards_memory_ctx(monkeypatch):
    from services import rag_service
    from rag.retriever import RetrievedDoc

    captured = {}

    class FakeRetriever:
        ready = True

        def retrieve(self, query, memory=None, exclude_ids=None,
                     q_vec=None):
            captured["query"] = query
            captured["topics"] = memory.last_topics()
            captured["followup"] = memory.looks_like_followup(query)
            captured["seen"] = memory.recently_seen_chunk_ids()
            result = type(
                "R",
                (),
                {
                    "docs": [],
                    "query_used": query,
                    "elapsed_s": 0.001,
                    "best_sim": 0.83,
                },
            )()
            result.docs = [
                RetrievedDoc(chunk_id="c1", path="p", text="t", score=0.9)
            ]
            return result

    monkeypatch.setattr(rag_service, "_retriever", FakeRetriever())
    monkeypatch.setattr(rag_service, "_situations", None)

    client = fastapi_testclient.TestClient(rag_service.app)
    resp = client.post(
        "/retrieve",
        json={
            "query": "nó ở đâu vậy?",
            "memory_ctx": {
                "topics": "đèn ông sao",
                "looks_like_followup": True,
                "seen_chunk_ids": ["c7"],
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"][0]["chunk_id"] == "c1"
    assert body["best_sim"] == 0.83
    assert captured["topics"] == "đèn ông sao"
    assert captured["followup"] is True
    assert captured["seen"] == {"c7"}


def test_rag_health_reports_error_when_not_initialized(monkeypatch):
    from services import rag_service

    monkeypatch.setattr(rag_service, "_retriever", None, raising=False)
    monkeypatch.setattr(rag_service, "_embedder", None, raising=False)
    client = fastapi_testclient.TestClient(rag_service.app)
    data = client.get("/health").json()
    assert data["status"] in ("loading", "error")


# ----------------------------------------------------------------------
# LLM service: SSE events are JSON-framed (newline-safe).
# ----------------------------------------------------------------------
def test_llm_stream_json_framing(monkeypatch):
    import json as _json

    from services import llm_service

    class FakeBackend:
        name = "fake"
        last_tool_events: list = []
        tool_skipped = True

        def stream(self, messages, temperature=None, max_tokens=None,
                   tools=False, memory_ctx=None, tool_executor=None,
                   force_search=True):
            yield "d\u1ee5ng m\u1ed9t\n"
            yield "d\u1ee5ng hai"

    monkeypatch.setattr(llm_service, "_backend", FakeBackend())
    client = fastapi_testclient.TestClient(llm_service.app)
    with client.stream(
        "POST",
        "/stream",
        json={"messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        payload = [
            _json.loads(line[6:])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]
    texts = [e.get("t") for e in payload if "t" in e]
    assert texts == ["d\u1ee5ng m\u1ed9t\n", "d\u1ee5ng hai"]
    assert payload[-1] == {"done": True, "tools": [], "skipped": True}


def test_llm_health_is_cheap(monkeypatch):
    from services import llm_service

    calls = {"n": 0}

    class FakeBackend:
        name = "fake"

        def health_check(self):
            calls["n"] += 1
            return True

    monkeypatch.setattr(llm_service, "_backend", FakeBackend())
    client = fastapi_testclient.TestClient(llm_service.app)
    assert client.get("/health").json()["status"] == "ok"
    assert calls["n"] == 0  # default poll costs NO generation
    client.get("/health", params={"deep": "1"})
    assert calls["n"] == 1  # deep probe does


# ----------------------------------------------------------------------
# TTS service: pure synthesis round-trip, cached flag, error passthrough.
# ----------------------------------------------------------------------
def test_tts_synthesize_roundtrip(monkeypatch):
    from services import tts_service

    pcm = np.array([5, -5, 1000], dtype=np.int16)

    class FakePlayer:
        disabled = False
        engine_name = "fake-tts"
        _cache_ns = "fake|voice"

        def __init__(self):
            self._cache = {}

        def _synthesize(self, text):
            key = f"{self._cache_ns}|{text}"
            self._cache[key] = (pcm, 24000)
            return self._cache[key]

    player = FakePlayer()
    monkeypatch.setattr(tts_service, "_player", player)
    monkeypatch.setattr(tts_service, "_init_error", "")
    client = fastapi_testclient.TestClient(tts_service.app)

    first = client.post("/synthesize", json={"text": "chào"}).json()
    assert first["cached"] is False
    decoded = np.frombuffer(base64.b64decode(first["audio_b64"]), dtype=np.int16)
    assert np.array_equal(decoded, pcm)
    assert first["sample_rate"] == 24000

    second = client.post("/synthesize", json={"text": "chào"}).json()
    assert second["cached"] is True


def test_tts_synthesize_reports_error_payload(monkeypatch):
    from services import tts_service

    class BrokenPlayer:
        disabled = False
        engine_name = "broken"
        _cache_ns = "b|v"
        _cache = {}

        def _synthesize(self, text):
            raise RuntimeError("weights missing")

    monkeypatch.setattr(tts_service, "_player", BrokenPlayer())
    monkeypatch.setattr(tts_service, "_init_error", "")
    client = fastapi_testclient.TestClient(tts_service.app)
    body = client.post("/synthesize", json={"text": "x"}).json()
    assert body["audio_b64"] is None and "weights missing" in body["error"]


# ----------------------------------------------------------------------
# ASR service: b64 float32 decode + not-ready behavior.
# ----------------------------------------------------------------------
def test_asr_transcribe_roundtrip(monkeypatch):
    from services import asr_service

    audio = np.full(320, 0.25, dtype=np.float32)

    class FakeSTT:
        ready = True

        def transcribe(self, a, sr):
            assert a.dtype == np.float32 and a.size == 320
            return "xin chào"

    monkeypatch.setattr(asr_service, "_stt", FakeSTT())
    client = fastapi_testclient.TestClient(asr_service.app)
    body = client.post(
        "/transcribe",
        json={
            "audio_b64": base64.b64encode(audio.tobytes()).decode(),
            "sample_rate": 16000,
        },
    ).json()
    assert body["text"] == "xin chào"


def test_asr_transcribe_when_not_ready_returns_empty(monkeypatch):
    from services import asr_service

    monkeypatch.setattr(asr_service, "_stt", None)
    client = fastapi_testclient.TestClient(asr_service.app)
    body = client.post(
        "/transcribe",
        json={"audio_b64": base64.b64encode(b"\x00\x00").decode()},
    ).json()
    assert body == {"text": "", "elapsed_ms": 0}
