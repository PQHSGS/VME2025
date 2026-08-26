"""TTS microservice — pure text->PCM synthesis, never touches a speaker.

Runs on port 8004. The kiosk process keeps playback/barge-in/bookkeeping
(TTSPlayer with its queue threads); this service only synthesizes and caches
PCM so the heavy VieNeu weights survive controller restarts.

Endpoints:
  POST /synthesize  — text -> base64 int16 PCM @ 24 kHz (cache-aware)
  POST /prewarm     — batch-prewarm filler/fallback lines into the cache
  GET  /health      — engine readiness (cheap; no synthesis)
  POST /reload      — hot-reload: rebuild the synth chain from current code
"""

from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from services.common import bootstrap, init_in_background

logger, cfg = bootstrap("tts_service")

app = FastAPI(title="TTS Service")
_player = None
_init_error = ""


class SynthesizeRequest(BaseModel):
    text: str


class SynthesizeResponse(BaseModel):
    audio_b64: str | None = None  # base64 int16 mono PCM @ sample_rate
    sample_rate: int = 0
    engine: str = ""
    elapsed_ms: int = 0
    cached: bool = False
    error: str = ""


class PrewarmRequest(BaseModel):
    phrases: list[str]


def _init_tts():
    global _player, _init_error
    _init_error = ""
    from tts import build_tts_player

    try:
        # probe=False: keep startup cheap; a broken engine surfaces on the
        # first /synthesize as an error field instead of killing the service.
        _player = build_tts_player(cfg, probe=False)
        logger.info(
            "TTS service initialized: %s", _player.engine_name if _player else "unknown"
        )
    except Exception as exc:
        logger.exception("TTS init failed")
        _player = None
        _init_error = str(exc)


@app.on_event("startup")
def startup():
    _init_tts()

    def warm():
        if _player is None or _player.disabled:
            return
        try:
            # Warm VieNeu weights so the first real synthesis doesn't pay a
            # multi-minute cold load mid-show.
            _player._synthesize("Xin chào các em nhỏ!")
            logger.info("TTS engine warmed")
        except Exception as exc:
            logger.warning("TTS warmup failed (engine falls back per-call): %s", exc)

    init_in_background(warm, "tts-warm")


def _cache_key(text: str) -> str:
    return f"{_player._cache_ns}|{text.strip()}"


@app.get("/health")
def health():
    if _player is None:
        return {"status": "error" if _init_error else "loading",
                "detail": _init_error}
    return {
        "status": "ok" if not _player.disabled else "disabled",
        "engine": _player.engine_name if _player else "unknown",
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    """Pure synthesis via TTSPlayer._synthesize: cache -> synth -> resample
    to 24 kHz -> LRU store. Never queues playback."""
    if _player is None:
        return SynthesizeResponse(error=_init_error or "player not ready")
    if _player.disabled:
        return SynthesizeResponse(error="tts disabled")
    started = time.perf_counter()
    key = _cache_key(req.text)
    cached = key in _player._cache
    try:
        # Runs in FastAPI's threadpool; blocks this one request only.
        pcm, rate = _player._synthesize(req.text)
    except Exception as exc:
        logger.warning("synthesis failed for %r: %s", req.text[:40], exc)
        return SynthesizeResponse(error=str(exc))
    return SynthesizeResponse(
        audio_b64=base64.b64encode(np.ascontiguousarray(pcm).tobytes()).decode(),
        sample_rate=rate,
        engine=_player.engine_name if _player else "",
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        cached=cached,
    )


@app.post("/prewarm")
def prewarm(req: PrewarmRequest):
    if _player is None or _player.disabled:
        return {"status": "disabled"}
    warmed = _player.prewarm(req.phrases)
    return {"status": "ok", "warmed": warmed}


@app.post("/reload")
def reload_model():
    global _player
    try:
        import importlib

        import tts

        importlib.reload(tts)
        _player = tts.build_tts_player(cfg, probe=False)
        return {
            "status": "reloaded",
            "engine": _player.engine_name if _player else "unknown",
        }
    except Exception as exc:
        logger.exception("reload failed")
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
