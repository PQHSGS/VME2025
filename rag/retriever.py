"""Retriever: cheap gate -> FAISS search -> MMR -> char budget.

Per turn:
  1. ``should_retrieve`` decides *whether* to pay for retrieval at all
     (keyword hits or centroid similarity) - most small talk skips it.
  2. Short conversational follow-ups are enriched with recent user topics
     before embedding ("nó ở đâu vậy?" keeps working).
  3. Candidates are re-ranked with MMR for diversity and trimmed to a hard
     character budget so the prompt never balloons.
"""

from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import FAISS_DIR

logger = logging.getLogger("rag.retriever")


@dataclass
class RetrievedDoc:
    chunk_id: str
    path: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    docs: list[RetrievedDoc] = field(default_factory=list)
    # The (possibly follow-up-enriched) text actually embedded. Equal to the
    # raw query when the question was self-contained - consumers use that to
    # decide whether a turn may enter the answer cache.
    query_used: str = ""
    elapsed_s: float = 0.0


class Retriever:
    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index = None
        self.docs_meta: list[dict] = []
        self.centroids: np.ndarray | None = None
        # LRU of query vectors (masks the gate -> retrieve double encode).
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # KB chunk vectors are static; encode each chunk once, ever.
        self._mmr_vec_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def load(self, index_dir: Path | None = None) -> bool:
        import faiss  # lazy heavy import

        index_dir = index_dir or FAISS_DIR
        try:
            self.index = faiss.read_index(str(index_dir / "index.faiss"))
            payload = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
            self.docs_meta = payload["docs"]
            centroid_path = index_dir / "centroids.npy"
            self.centroids = np.load(centroid_path) if centroid_path.exists() else None
            logger.info(
                "retriever loaded %d chunks from %s", len(self.docs_meta), index_dir
            )
            return True
        except (OSError, KeyError, ValueError) as exc:
            logger.error("failed to load index from %s: %s", index_dir, exc)
            return False

    @property
    def ready(self) -> bool:
        return self.index is not None

    def warm_vectors(self):
        """Pre-encode every KB chunk in a daemon thread.

        First-touch MMR otherwise costs ~8 embed calls (~2-3s CPU) for the
        first visitor who asks about an untouched section. Returns the
        warming thread so callers/benches can join it.
        """
        import threading

        def run():
            started = time.perf_counter()
            for meta in self.docs_meta:
                self._chunk_vector(meta)
            logger.info(
                "mmr vector cache warmed: %d chunks in %.1fs",
                len(self._mmr_vec_cache),
                time.perf_counter() - started,
            )

        thread = threading.Thread(target=run, name="mmr-warm", daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------
    _EMBED_CACHE_MAX = 512

    def _embed_query(self, text: str) -> np.ndarray:
        cached = self._embed_cache.get(text)
        if cached is not None:
            self._embed_cache.move_to_end(text)
            return cached
        vec = self.embedder.encode_query(text)
        if len(self._embed_cache) >= self._EMBED_CACHE_MAX:
            self._embed_cache.popitem(last=False)  # evict coldest entry
        self._embed_cache[text] = vec
        return vec

    # ------------------------------------------------------------------
    def should_retrieve(self, query: str, memory=None) -> bool:
        q = query.strip().lower()
        if not q:
            return False
        if any(keyword in q for keyword in self.cfg.domain_keywords):
            return True
        if self.centroids is not None and len(q.split()) >= 3:
            vec = self._embed_query(query)
            sims = self.centroids @ vec
            if float(sims.max()) >= self.cfg.gate_threshold:
                return True
        return False

    def _should_retrieve_vec(self, query: str, q_vec: np.ndarray, memory=None) -> bool:
        """Gate check using a pre-computed query vector (avoids re-encoding)."""
        q = query.strip().lower()
        if not q:
            return False
        if any(keyword in q for keyword in self.cfg.domain_keywords):
            return True
        if self.centroids is not None and len(q.split()) >= 3:
            sims = self.centroids @ q_vec
            if float(sims.max()) >= self.cfg.gate_threshold:
                return True
        return False

    def _effective_query(self, query: str, memory=None) -> str:
        """Enrich very short follow-ups with recent topics."""
        words = query.strip().split()
        if memory is not None and len(words) <= 5 and memory.looks_like_followup(query):
            topics = memory.last_topics()
            if topics:
                enriched = f"{topics} {query}".strip()
                logger.debug("query enriched: %r -> %r", query, enriched)
                return enriched
        return query

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        memory=None,
        exclude_ids: set[str] | None = None,
        force: bool = False,
        q_vec: np.ndarray | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        result = RetrievalResult(query_used=query)

        if not self.ready:
            logger.warning("retriever called before index load")
            return result

        # Enrich follow-ups first, then embed once for gate + FAISS.
        effective_query = self._effective_query(query, memory)
        if q_vec is None:
            q_vec = self._embed_query(effective_query)
        elif effective_query != query:
            # Follow-up enriched the text; encode the enriched version.
            q_vec = self._embed_query(effective_query)

        if not force and not self._should_retrieve_vec(query, q_vec, memory):
            result.elapsed_s = time.perf_counter() - started
            return result

        k = min(self.cfg.retriever_topk_candidates, self.index.ntotal)  # type: ignore[union-attr]
        scores, ids = self.index.search(q_vec[None, :].astype(np.float32), k)  # type: ignore[union-attr]
        candidates: list[tuple[dict, float]] = []
        seen_turn_window = set()
        if memory is not None:
            seen_turn_window = memory.recently_seen_chunk_ids(
                self.cfg.dedup_window_turns
            )
        penalty_applied: dict[str, float] = {}
        for raw_score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            score = float(raw_score)
            meta = self.docs_meta[int(idx)]
            if meta["chunk_id"] in seen_turn_window:
                penalty_applied[meta["chunk_id"]] = self.cfg.dedup_penalty
                score -= self.cfg.dedup_penalty
            if meta["chunk_id"] in (exclude_ids or set()):
                continue
            if score < self.cfg.retriever_min_score:
                continue
            candidates.append((meta, score))

        if not candidates:
            result.elapsed_s = time.perf_counter() - started
            return result

        picked = self._mmr(q_vec, candidates, self.cfg.retriever_final_docs)
        picked = self._apply_budget(picked, self.cfg.context_char_budget)
        result.docs = [
            RetrievedDoc(
                chunk_id=m["chunk_id"], path=m["path"], text=m["text"], score=s
            )
            for m, s in picked
        ]
        result.elapsed_s = time.perf_counter() - started
        logger.info(
            "retrieved %d docs in %.3fs (q=%r%s)",
            len(result.docs),
            result.elapsed_s,
            effective_query[:60],
            f", penalties={list(penalty_applied)}" if penalty_applied else "",
        )
        return result

    # ------------------------------------------------------------------
    def _mmr(
        self, q_vec: np.ndarray, candidates: list[tuple[dict, float]], top_n: int
    ) -> list[tuple[dict, float]]:
        """Maximal Marginal Relevance over candidate embeddings."""
        lam = self.cfg.retriever_mmr_lambda
        metas = [m for m, _ in candidates]
        vectors = np.stack([self._chunk_vector(m) for m in metas])
        relevance = np.array([s for _, s in candidates], dtype=np.float32)
        selected: list[int] = []
        remaining = list(range(len(metas)))
        while remaining and len(selected) < top_n:
            if not selected:
                best = int(np.argmax(relevance[remaining]))
            else:
                sim_to_selected = vectors[remaining] @ vectors[selected].T
                max_sim = sim_to_selected.max(axis=1)
                mmr_scores = lam * relevance[remaining] - (1 - lam) * max_sim
                best = int(np.argmax(mmr_scores))
            selected.append(remaining.pop(best))
        return [(metas[i], float(relevance[i])) for i in selected]

    def _chunk_vector(self, meta: dict) -> np.ndarray:
        """Embedding of a static KB chunk - computed once per chunk_id."""
        cid = meta["chunk_id"]
        vec = self._mmr_vec_cache.get(cid)
        if vec is None:
            text = f"[{meta['path']}] {meta['text']}"
            (vec,) = self.embedder.encode([text], normalize=True)
            self._mmr_vec_cache[cid] = vec
        return vec

    def _apply_budget(
        self, picked: list[tuple[dict, float]], char_budget: int
    ) -> list[tuple[dict, float]]:
        total = sum(len(m["text"]) for m, _ in picked)
        while len(picked) > 1 and total > char_budget:
            dropped = picked.pop()  # lowest MMR rank first
            total -= len(dropped[0]["text"])
        return picked
