"""ASR microservice — gipformer-65M int8 via sherpa-onnx.

Runs on port 8001. Accepts base64-encoded float32 audio, returns transcript.

Endpoints:
  POST /transcribe  — audio bytes -> text
  GET  /health      — model ready check
  POST /reload      — hot-reload: re-instantiate the ASR model
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config, GIPFORMER_DIR, SAMPLE_RATE

logger = logging.getLogger("asr_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ASR Service")
cfg = Config()
_stt = None


class TranscribeRequest(BaseModel):
    audio_b64: str  # base64-encoded float32 numpy array
    sample_rate: int = SAMPLE_RATE


class TranscribeResponse(BaseModel):
    text: str
    elapsed_ms: int


def _load_model():
    global _stt
    from asr import GipformerSTT

    _stt = GipformerSTT(cfg)
    _stt.load()
    logger.info("ASR model loaded")


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health")
def health():
    return {
        "status": "ok" if _stt and _stt.ready else "loading",
        "backend": "gipformer-65M-i8",
    }


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
    """Hot-reload: re-instantiate the ASR model from potentially changed code."""
    global _stt
    try:
        import importlib
        import asr

        importlib.reload(asr)
        _stt = asr.GipformerSTT(cfg)
        _stt.load()
        return {"status": "reloaded", "backend": "gipformer-65M-i8"}
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
