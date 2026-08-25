"""RAG microservice — FAISS retrieval + embedding.

Runs on port 8003. Handles query encoding, gate check, FAISS search, MMR,
and situation matching. All heavy state (index, embedder, situation vectors)
lives in this process and survives LLM/TTS restarts.

Endpoints:
  POST /retrieve    — query -> retrieved docs
  POST /situation   — query -> situation match or null
  POST /embed       — text -> query vector (for other services)
  GET  /health      — index + embedder ready check
  POST /reload      — hot-reload: re-init embedder + index
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config

logger = logging.getLogger("rag_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Service")
cfg = Config()
_embedder = None
_retriever = None
_situations = None


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    vector: list[float]
    dim: int


class RetrieveRequest(BaseModel):
    query: str
    q_vec: list[float] | None = None
    force: bool = False


class RetrievedDoc(BaseModel):
    chunk_id: str
    path: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    docs: list[RetrievedDoc]
    query_used: str
    elapsed_ms: int


class SituationRequest(BaseModel):
    query: str
    q_vec: list[float] | None = None


class SituationResponse(BaseModel):
    matched: bool
    question: str | None = None
    answer: str | None = None
    score: float | None = None


def _init_rag():
    global _embedder, _retriever, _situations
    from rag.embedder import SentenceTransformersEmbedder
    from rag.retriever import Retriever
    from rag.situations import SituationMatcher

    _embedder = SentenceTransformersEmbedder(
        cfg.embed_model,
        query_prompt=cfg.embed_query_prompt,
        num_threads=cfg.embed_threads,
    )
    _retriever = Retriever(cfg, _embedder)
    _retriever.load()
    _retriever.warm_vectors()
    _situations = SituationMatcher(cfg, _embedder)
    _situations.load()
    logger.info("RAG service initialized")


@app.on_event("startup")
def startup():
    _init_rag()


@app.get("/health")
def health():
    return {
        "status": "ok" if _retriever and _retriever.ready else "loading",
        "index_chunks": _retriever.index.ntotal if _retriever and _retriever.ready else 0,
        "situations_rows": len(_situations.rows) if _situations else 0,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if _embedder is None:
        return EmbedResponse(vector=[], dim=0)
    vec = _embedder.encode_query(req.text)
    return EmbedResponse(vector=vec.tolist(), dim=len(vec))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if _retriever is None or not _retriever.ready:
        return RetrieveResponse(docs=[], query_used=req.query, elapsed_ms=0)
    q_vec = np.array(req.q_vec, dtype=np.float32) if req.q_vec else None
    result = _retriever.retrieve(req.query, q_vec=q_vec, force=req.force)
    docs = [
        RetrievedDoc(
            chunk_id=d.chunk_id, path=d.path, text=d.text, score=d.score
        )
        for d in result.docs
    ]
    return RetrieveResponse(
        docs=docs,
        query_used=result.query_used,
        elapsed_ms=int(result.elapsed_s * 1000),
    )


@app.post("/situation", response_model=SituationResponse)
def situation(req: SituationRequest):
    if _situations is None or not _situations.ready:
        return SituationResponse(matched=False)
    q_vec = np.array(req.q_vec, dtype=np.float32) if req.q_vec else None
    hit = _situations.match(req.query, q_vec=q_vec)
    if hit is None:
        return SituationResponse(matched=False)
    return SituationResponse(
        matched=True,
        question=hit.question,
        answer=hit.answer,
        score=hit.score,
    )


@app.post("/reload")
def reload_model():
    global _embedder, _retriever, _situations
    try:
        import importlib
        import rag.embedder
        import rag.retriever
        import rag.situations

        importlib.reload(rag.embedder)
        importlib.reload(rag.retriever)
        importlib.reload(rag.situations)
        _init_rag()
        return {"status": "reloaded"}
    except Exception as exc:
        logger.exception("reload failed")
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
