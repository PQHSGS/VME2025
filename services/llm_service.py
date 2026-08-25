"""LLM microservice — Gemini via google-genai SDK.

Runs on port 8002. Proxies streaming and non-streaming LLM calls.

Endpoints:
  POST /stream      — messages -> token stream (SSE)
  POST /complete    — messages -> full text
  GET  /health      — API key + connectivity check
  POST /reload      — hot-reload: re-init the backend
"""

from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.common import bootstrap, init_in_background

logger, cfg = bootstrap("llm_service")

app = FastAPI(title="LLM Service")
_backend = None
_init_error = ""


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
    global _backend, _init_error
    _init_error = ""
    try:
        from llm import select_backend

        _backend = select_backend(cfg)
        logger.info("LLM backend initialized: %s", _backend.name)
    except Exception as exc:
        logger.error("LLM backend init failed: %s", exc)
        _backend = None
        _init_error = str(exc)


@app.on_event("startup")
def startup():
    init_in_background(_init_backend, "llm-init")


@app.get("/health")
def health(deep: bool = False):
    """Cheap readiness by default: startup's select_backend already did a
    real generation probe, so polling here must not cost a Gemini call.
    Pass ?deep=1 for a live end-to-end check."""
    if _backend is None:
        return {"status": "error" if _init_error else "loading",
                "detail": _init_error}
    if deep:
        ok = _backend.health_check()
        return {"status": "ok" if ok else "unhealthy", "backend": _backend.name}
    return {"status": "ok", "backend": _backend.name}


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
        # JSON-encode every event: raw token chunks may contain "\n" which
        # would break SSE framing ("data: <chunk with \n>" splits wrongly).
        for chunk in _backend.stream(
            messages, temperature=req.temperature, max_tokens=req.max_tokens
        ):
            yield f"data: {json.dumps({'t': chunk}, ensure_ascii=False)}\n\n"
        yield 'data: {"done": true}\n\n'

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
