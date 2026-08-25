"""ASR microservice — gipformer-65M int8 via sherpa-onnx (whisper legacy).

Runs on port 8001. Accepts base64-encoded float32 audio, returns transcript.

Startup never crashes the process: a missing model dir (e.g. models/
gipformer-65M-i8 not yet copied onto this box) surfaces in /health as
status="error" and POST /reload retries once the files appear.

Endpoints:
  POST /transcribe  — audio bytes -> text
  GET  /health      — model ready check
  POST /reload      — hot-reload: re-instantiate the ASR model
"""

from __future__ import annotations

import base64
import threading
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from services.common import bootstrap, init_in_background
from config import SAMPLE_RATE

logger, cfg = bootstrap("asr_service")

app = FastAPI(title="ASR Service")
_stt = None
_init_error = ""
_load_lock = threading.Lock()


class TranscribeRequest(BaseModel):
    audio_b64: str  # base64-encoded float32 mono samples
    sample_rate: int = SAMPLE_RATE


class TranscribeResponse(BaseModel):
    text: str
    elapsed_ms: int


def _build_stt():
    from asr import GipformerSTT, WhisperSTT

    return GipformerSTT(cfg) if cfg.asr_backend == "gipformer" else WhisperSTT(cfg)


def _load_model():
    global _stt, _init_error
    with _load_lock:
        if _stt is not None and _stt.ready:
            return True
        try:
            stt = _build_stt()
            stt.load()  # raises on missing/broken model files
            _stt, _init_error = stt, ""
            logger.info("ASR model loaded (%s)", cfg.asr_backend)
            return True
        except Exception as exc:
            logger.error("ASR load failed: %s", exc)
            _stt, _init_error = None, str(exc)
            return False


@app.on_event("startup")
def startup():
    init_in_background(_load_model, "asr-load")


@app.get("/health")
def health():
    if _stt is not None and _stt.ready:
        return {"status": "ok", "backend": cfg.asr_backend}
    if _init_error:
        return {"status": "error", "detail": _init_error}
    return {"status": "loading", "backend": cfg.asr_backend}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    if not _stt or not _stt.ready:
        return TranscribeResponse(text="", elapsed_ms=0)
    started = time.perf_counter()
    audio_bytes = base64.b64decode(req.audio_b64)
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    text = _stt.transcribe(audio, req.sample_rate)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return TranscribeResponse(text=text, elapsed_ms=elapsed_ms)


@app.post("/reload")
def reload_model():
    """Hot-reload current code + retry loading from disk."""
    global _stt
    try:
        import importlib

        import asr

        with _load_lock:
            _stt = None
            importlib.reload(asr)
        ok = _load_model()
        return {
            "status": "reloaded" if ok else "error",
            "detail": _init_error,
            "backend": cfg.asr_backend,
        }
    except Exception as exc:
        logger.exception("reload failed")
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
