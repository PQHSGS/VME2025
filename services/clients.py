"""Remote stand-ins that match the local component protocols exactly.

Microservice mode injects these into ConversationOrchestrator so the
production code path is unchanged (invariant: everything is injectable).
Each class speaks HTTP to one service but exposes the same attributes and
methods its local twin has — including the quirks the orchestrator relies
on (``retriever.embedder``, ``result.query_used``, raising LLM backends for
the circuit breaker, synth failures counting toward TTS auto-disable).

Design notes:
  * ONE shared keepalive httpx.Client per process — connection reuse keeps
    the per-hop overhead at ~1-5ms on loopback.
  * Readiness is cached briefly (``ready`` re-probes /health at most every
    READY_TTL_S) so hot-path checks never pay a health round-trip.
  * Failures follow local semantics: retrieval/situations degrade to empty,
    ASR returns "", TTS synth raises (TTSPlayer disables after 3), LLM
    raises (orchestrator's FailureTracker trips the breaker).
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time

import httpx
import numpy as np

logger = logging.getLogger("services.clients")

_READY_TTL_S = 5.0


def get_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Shared keepalive client; tests may inject a MockTransport once."""
    global _SHARED_CLIENT
    if transport is not None:
        return httpx.Client(transport=transport)
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=8)
        )
    return _SHARED_CLIENT


_SHARED_CLIENT: httpx.Client | None = None


def close_shared_client() -> None:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None:
        _SHARED_CLIENT.close()
        _SHARED_CLIENT = None


def _timeout(read_s: float) -> httpx.Timeout:
    # connect must stay tight: a dead service should fail fast, not hang the
    # turn. read covers cold model loads on first call (VieNeu ~tens of s).
    return httpx.Timeout(connect=2.0, read=read_s, write=30.0, pool=5.0)


class _HealthCache:
    """Brief-cached readiness probe shared by all remote stand-ins."""

    def __init__(self, url: str, transport: httpx.BaseTransport | None = None):
        self._url = url.rstrip("/")
        self._transport = transport
        self._ok_until = 0.0
        self._detail = ""

    def refresh(self, timeout: float = 3.0) -> bool:
        try:
            resp = get_client(self._transport).get(
                f"{self._url}/health", timeout=timeout
            )
            data = resp.json()
            ok = data.get("status") == "ok"
            self._detail = str(data.get("detail", data.get("engine", "")))
            self._ok_until = time.monotonic() + (_READY_TTL_S if ok else 0.0)
            return ok
        except Exception as exc:
            self._detail = str(exc)
            self._ok_until = 0.0
            return False

    @property
    def ready(self) -> bool:
        if time.monotonic() < self._ok_until:
            return True
        return self.refresh()


# ----------------------------------------------------------------------
class RemoteEmbedder:
    """Embeds via RAG service /embed — keeps embed-once semantics working.

    The controller fetches q_vec ONCE per turn and passes it to situations /
    retriever / answer cache, so the heavy model runs in exactly one process.
    """

    dim = 1024  # corrected after first response

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.base = cfg.rag_service_url.rstrip("/")
        self._transport = transport

    def encode_query(self, text: str) -> np.ndarray:
        resp = get_client(self._transport).post(
            f"{self.base}/embed", json={"text": text}, timeout=_timeout(60.0)
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("vector"):
            raise RuntimeError("rag service returned empty embedding")
        self.dim = int(data["dim"])
        return np.asarray(data["vector"], dtype=np.float32)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        # Fallback path only: the orchestrator embeds once per turn via
        # encode_query and shares q_vec with every consumer.
        return np.stack([self.encode_query(t) for t in texts]) if texts else (
            np.zeros((0, self.dim), dtype=np.float32)
        )


# ----------------------------------------------------------------------
class RemoteRetriever:
    """Drop-in for rag.retriever.Retriever over HTTP."""

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.base = cfg.rag_service_url.rstrip("/")
        self._transport = transport
        self.embedder = RemoteEmbedder(cfg, transport)
        self.health = _HealthCache(self.base, transport)

    @property
    def ready(self) -> bool:
        return self.health.ready

    def load(self, index_dir=None) -> bool:
        return self.health.refresh()

    def warm_vectors(self):
        return None  # server warms its own MMR cache at startup

    def retrieve(self, query, memory=None, exclude_ids=None, force=False, q_vec=None):
        from rag.retriever import RetrievalResult

        if not self.ready:
            logger.debug("rag service not ready - empty retrieval")
            return RetrievalResult(query_used=query)
        payload: dict = {"query": query, "force": force}
        if q_vec is not None:
            payload["q_vec"] = np.asarray(q_vec, dtype=np.float32).tolist()
        if memory is not None:
            payload["memory_ctx"] = {
                "topics": memory.last_topics(),
                "looks_like_followup": memory.looks_like_followup(query),
                "seen_chunk_ids": sorted(
                    memory.recently_seen_chunk_ids(self.cfg.dedup_window_turns)
                ),
            }
        try:
            resp = get_client(self._transport).post(
                f"{self.base}/retrieve", json=payload, timeout=_timeout(60.0)
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("retrieve failed (%s) - degrading to no docs", exc)
            return RetrievalResult(query_used=query)
        result = RetrievalResult(query_used=data.get("query_used", query))
        from rag.retriever import RetrievedDoc

        result.docs = [
            RetrievedDoc(
                chunk_id=d["chunk_id"],
                path=d["path"],
                text=d["text"],
                score=float(d["score"]),
            )
            for d in data.get("docs", [])
        ]
        return result


# ----------------------------------------------------------------------
class RemoteSituations:
    """Drop-in for rag.situations.SituationMatcher over HTTP."""

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.base = cfg.rag_service_url.rstrip("/")
        self._transport = transport
        self.embedder = RemoteEmbedder(cfg, transport)
        self.rows: list[dict] = []
        self.health = _HealthCache(self.base, transport)

    def load(self) -> bool:
        try:
            resp = get_client(self._transport).get(
                f"{self.base}/situations", timeout=_timeout(30.0)
            )
            resp.raise_for_status()
            data = resp.json()
            self.rows = data.get("rows", [])
            enabled = bool(data.get("enabled"))
            return bool(self.rows) and enabled and self.health.refresh()
        except Exception as exc:
            logger.warning("situations load failed: %s", exc)
            self.rows = []
            return False

    @property
    def ready(self) -> bool:
        return bool(self.rows) and self.cfg.situations_enabled and self.health.ready

    def match(self, query: str, q_vec=None):
        from rag.situations import Situation

        if not self.rows or not self.cfg.situations_enabled:
            return None
        payload: dict = {"query": query}
        if q_vec is not None:
            payload["q_vec"] = np.asarray(q_vec, dtype=np.float32).tolist()
        try:
            resp = get_client(self._transport).post(
                f"{self.base}/situation", json=payload, timeout=_timeout(15.0)
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("situation match failed: %s", exc)
            return None
        if not data.get("matched"):
            return None
        row = {
            "question": data["question"],
            "guidance": "",
            "answer": data["answer"],
        }
        return Situation(score=float(data["score"]), **row)


# ----------------------------------------------------------------------
class RemoteSTT:
    """Drop-in for GipformerSTT/WhisperSTT over HTTP."""

    name = "asr-service"

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.base = cfg.asr_service_url.rstrip("/")
        self._transport = transport
        self.health = _HealthCache(self.base, transport)

    @property
    def ready(self) -> bool:
        return self.health.ready

    def load_async(self, callback=None) -> threading.Thread:
        def worker():
            deadline = time.monotonic() + 60.0
            ok = False
            while time.monotonic() < deadline:
                if self.health.refresh():
                    ok = True
                    break
                time.sleep(0.5)
            if callback:
                callback(ok)

        thread = threading.Thread(target=worker, name="asr-remote-load", daemon=True)
        thread.start()
        return thread

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        payload = {
            "audio_b64": base64.b64encode(
                np.ascontiguousarray(audio, dtype=np.float32).tobytes()
            ).decode(),
            "sample_rate": sample_rate,
        }
        try:
            resp = get_client(self._transport).post(
                f"{self.base}/transcribe", json=payload, timeout=_timeout(30.0)
            )
            resp.raise_for_status()
            text = resp.json().get("text", "")
        except Exception as exc:
            # Matches monolith "model not loaded" behavior: drop the utterance.
            logger.warning("remote transcribe failed: %s", exc)
            return ""
        logger.info("transcribed via service: %r", text)
        return text


# ----------------------------------------------------------------------
class RemoteLLM:
    """Drop-in for llm.GeminiBackend over HTTP (SSE streaming).

    Raises on transport/service errors so the orchestrator's FailureTracker
    + circuit breaker behave identically to the in-process backend.
    """

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.name = "llm-service"
        self.base = cfg.llm_service_url.rstrip("/")
        self._transport = transport

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with get_client(self._transport).stream(
            "POST", f"{self.base}/stream", json=payload, timeout=_timeout(30.0)
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"llm service returned {resp.status_code}")
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("done"):
                    break
                token = event.get("t")
                if token:
                    yield token

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = get_client(self._transport).post(
            f"{self.base}/complete", json=payload, timeout=_timeout(60.0)
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    def health_check(self) -> bool:
        try:
            resp = get_client(self._transport).get(
                f"{self.base}/health",
                params={"deep": "1"},
                timeout=_timeout(20.0),
            )
            return resp.json().get("status") == "ok"
        except Exception:
            return False


# ----------------------------------------------------------------------
class RemoteSynth:
    """text -> (int16 pcm @24kHz) via TTS service; raises so TTSPlayer's
    consecutive-failure disable logic still guards the kiosk."""

    def __init__(self, cfg, transport: httpx.BaseTransport | None = None):
        self.base = cfg.tts_service_url.rstrip("/")
        self._transport = transport

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        # read=300: a cold VieNeu load inside the service can take minutes;
        # synthesis runs on TTSPlayer's worker thread so this never blocks
        # the kiosk main loop.
        resp = get_client(self._transport).post(
            f"{self.base}/synthesize", json={"text": text}, timeout=_timeout(300.0)
        )
        resp.raise_for_status()
        data = resp.json()
        audio_b64 = data.get("audio_b64")
        if not audio_b64:
            raise RuntimeError(data.get("error") or "empty synthesis")
        pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
        return pcm, int(data.get("sample_rate", 24000))


def build_remote_tts_player(cfg, transport: httpx.BaseTransport | None = None):
    """Local playback machinery + remote synthesis. Barge-in bookkeeping,
    heard-text fidelity and idle device handling all stay in this process."""
    from tts import TTSPlayer

    player = TTSPlayer(cfg, synth_pcm_fn=RemoteSynth(cfg, transport))
    logger.info("TTS engine: remote service (%s)", player.engine_name)
    return player
