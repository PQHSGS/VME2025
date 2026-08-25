"""LLM microservice — Gemini via google-genai SDK.

Runs on port 8002. Proxies streaming and non-streaming LLM calls.

Endpoints:
  POST /stream      — messages -> token stream (SSE)
  POST /complete    — messages -> full text
  GET  /health      — API key + connectivity check
  POST /reload      — hot-reload: re-init the backend
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Iterator

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config

logger = logging.getLogger("llm_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="LLM Service")
cfg = Config()
_backend = None


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


class GenerateResponse(BaseModel):
    text: str
    elapsed_ms: int


def _init_backend():
    global _backend
    from llm import select_backend

    _backend = select_backend(cfg)
    logger.info("LLM backend initialized: %s", _backend.name)


@app.on_event("startup")
def startup():
    _init_backend()


@app.get("/health")
def health():
    if _backend is None:
        return {"status": "loading"}
    ok = _backend.health_check()
    return {"status": "ok" if ok else "unhealthy", "backend": _backend.name}


@app.post("/complete", response_model=GenerateResponse)
def complete(req: GenerateRequest):
    if _backend is None:
        return GenerateResponse(text="", elapsed_ms=0)
    started = time.perf_counter()
    messages = [m.model_dump() for m in req.messages]
    text = _backend.complete(messages, temperature=req.temperature, max_tokens=req.max_tokens)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GenerateResponse(text=text, elapsed_ms=elapsed_ms)


@app.post("/stream")
def stream(req: GenerateRequest):
    if _backend is None:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    messages = [m.model_dump() for m in req.messages]

    def generate() -> Iterator[str]:
        for chunk in _backend.stream(
            messages, temperature=req.temperature, max_tokens=req.max_tokens
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reload")
def reload_model():
    global _backend
    try:
        import importlib
        import llm

        importlib.reload(llm)
        _backend = llm.select_backend(cfg)
        return {"status": "reloaded", "backend": _backend.name}
    except Exception as exc:
        logger.exception("reload failed")
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
