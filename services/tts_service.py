"""TTS microservice — VieNeu / edge-tts synthesis.

Runs on port 8004. Accepts text, returns audio bytes (PCM int16 or MP3).

Endpoints:
  POST /synthesize  — text -> audio bytes
  POST /prewarm     — batch-prewarm filler phrases
  GET  /health      — engine ready check
  POST /reload      — hot-reload: re-init TTS engine
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config

logger = logging.getLogger("tts_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="TTS Service")
cfg = Config()
_player = None


class SynthesizeRequest(BaseModel):
    text: str
    tag: str = "reply"


class SynthesizeResponse(BaseModel):
    audio_b64: str | None = None  # base64-encoded int16 PCM
    sample_rate: int = 0
    engine: str = ""
    elapsed_ms: int = 0
    cached: bool = False


class PrewarmRequest(BaseModel):
    phrases: list[str]


def _init_tts():
    global _player
    from tts import build_tts_player

    _player = build_tts_player(cfg, probe=False)
    logger.info("TTS service initialized: %s", getattr(_player, "engine_name", "unknown"))


@app.on_event("startup")
def startup():
    _init_tts()


@app.get("/health")
def health():
    if _player is None:
        return {"status": "loading"}
    return {
        "status": "ok" if not _player.disabled else "disabled",
        "engine": getattr(_player, "engine_name", "unknown"),
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if _player is None or _player.disabled:
        return SynthesizeResponse()
    started = time.perf_counter()
    # Check cache first
    cached = None
    if hasattr(_player, "_cache"):
        cached = _player._cache.get(req.text)
    if cached is not None:
        return SynthesizeResponse(
            audio_b64=base64.b64encode(cached).decode(),
            sample_rate=24000,
            engine=getattr(_player, "engine_name", ""),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            cached=True,
        )
    # Synthesize
    try:
        if hasattr(_player, "_synth_pcm") and _player._synth_pcm is not None:
            pcm, rate = _player._synth_pcm(req.text)
            audio_bytes = pcm.tobytes()
        else:
            # Fallback: submit to queue and wait (edge-tts path)
            _player.submit(req.text, tag=req.tag)
            return SynthesizeResponse(
                engine=getattr(_player, "engine_name", ""),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        return SynthesizeResponse(
            audio_b64=base64.b64encode(audio_bytes).decode(),
            sample_rate=rate,
            engine=getattr(_player, "engine_name", ""),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        logger.exception("synthesis failed")
        return SynthesizeResponse()


@app.post("/prewarm")
def prewarm(req: PrewarmRequest):
    if _player is None or _player.disabled:
        return {"status": "disabled"}
    prewarm_fn = getattr(_player, "prewarm", None)
    if callable(prewarm_fn):
        prewarm_fn(req.phrases)
    return {"status": "ok", "warmed": len(req.phrases)}


@app.post("/reload")
def reload_model():
    global _player
    try:
        import importlib
        import tts

        importlib.reload(tts)
        _player = tts.build_tts_player(cfg, probe=False)
        return {"status": "reloaded", "engine": getattr(_player, "engine_name", "unknown")}
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
