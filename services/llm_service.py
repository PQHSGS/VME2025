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
    # Tool mode: service-side search_kb loop against the RAG service. The
    # orchestrator ships a memory snapshot so retrieval keeps its dedup and
    # follow-up enrichment without cross-process object references.
    tools: bool = False
    memory_ctx: dict | None = None
    force_search: bool = True


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


_rag_client = None  # lazy RemoteRetriever shared across tool-mode requests


def _get_rag_client():
    global _rag_client
    if _rag_client is None:
        from services.clients import RemoteRetriever

        _rag_client = RemoteRetriever(cfg)
    return _rag_client


@app.post("/stream")
def stream(req: GenerateRequest):
    if _backend is None:
        return StreamingResponse(iter([]), media_type="text/event-stream")

    messages = [m.model_dump() for m in req.messages]
    memory_ctx = req.memory_ctx

    def generate() -> Iterator[str]:
        # JSON-encode every event: raw token chunks may contain "\n" which
        # would break SSE framing ("data: <chunk with \n>" splits wrongly).
        # Rich tool events ride inside the done event so the orchestrator
        # can trace docs/best_sim without another round-trip.
        tool_events: list[dict] = []
        rag_ready = True

        if req.tools:
            rag_ready = _get_rag_client().ready
            if not rag_ready:
                logger.warning("tool mode: RAG not ready - steering, not searching")

        def executor(query: str) -> str:
            from prompts import format_retrieved_block

            nonlocal rag_ready
            # Fail FAST while the fleet settles: a blocking retrieve against
            # a warming embedder would starve the SSE read window.
            if not rag_ready or not _get_rag_client().ready:
                return (
                    "KHÔNG tìm thấy tài liệu phù hợp. Hãy trả lời em nhí "
                    "bằng hiểu biết chung của Ông một cách thân thiện, và "
                    "không bịa chi tiết về bảo tàng."
                )
            result = _get_rag_client().retrieve(
                query, raw_memory_ctx=memory_ctx
            )
            result_docs = [
                {"path": d.path, "text": d.text, "score": d.score}
                for d in result.docs
            ]
            best_sim = result.best_sim
            logger.info(
                "tool search %r -> %d docs (best_sim=%.3f)",
                query[:60],
                len(result_docs),
                best_sim,
            )
            tool_events.append(
                {
                    "query": query[:120],
                    "docs": len(result_docs),
                    "best_sim": round(best_sim, 3),
                }
            )
            if not result_docs:
                # Steer leg 2 instead of inviting another tool round-trip.
                return (
                    "KHÔNG tìm thấy tài liệu phù hợp. Hãy trả lời em nhí "
                    "bằng hiểu biết chung của Ông một cách thân thiện, và "
                    "không bịa chi tiết về bảo tàng."
                )
            return format_retrieved_block(result_docs)[:8000]

        for chunk in _backend.stream(
            messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            tools=req.tools,
            memory_ctx=memory_ctx,
            tool_executor=executor if req.tools else None,
            force_search=req.force_search,
        ):
            yield f"data: {json.dumps({'t': chunk}, ensure_ascii=False)}\n\n"
        yield "data: " + json.dumps(
            {
                "done": True,
                "tools": tool_events or list(_backend.last_tool_events if _backend else []),
                "skipped": bool(_backend.tool_skipped if _backend else True),
            },
            ensure_ascii=False,
        ) + "\n\n"

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
