"""Retriever: FAISS search -> evidence bar -> MMR -> char budget.

Per turn:
  1. Always search the index (embedder is already paid for the turn).
  2. EVIDENCE BAR: docs are included only when the best raw cosine between
     the query and the KB clears ``evidence_sim_min``. Below the bar there
     is no docs block at all - the LLM continues from native history alone.
     This replaces intent-guessing heuristics: we measure whether knowledge
     exists instead of predicting utterance type.
  3. Short conversational follow-ups are enriched with recent topics before
     embedding ("nó ở đâu vậy?" keeps working).
  4. Candidates are re-ranked with MMR and trimmed to a char budget so the
     prompt never balloons.
"""

from __future__ import annotations

import json
import logging
import threading
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
    # Best raw cosine against the KB for this turn, regardless of whether
    # docs were included. Traces use it to tune EVIDENCE_SIM_MIN.
    best_sim: float = 0.0
    elapsed_s: float = 0.0


class Retriever:
    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index = None
        self.docs_meta: list[dict] = []
        # LRU of query vectors (dedupes repeated encodes within a process).
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # KB chunk vectors are static; encode each chunk once, ever.
        self._mmr_vec_cache: dict[str, np.ndarray] = {}
        self._vectors_warm = threading.Event()

    # ------------------------------------------------------------------
    def load(self, index_dir: Path | None = None) -> bool:
        import faiss  # lazy heavy import

        index_dir = index_dir or FAISS_DIR
        try:
            self.index = faiss.read_index(str(index_dir / "index.faiss"))
            payload = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
            self.docs_meta = payload["docs"]
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

    def warm_vectors(self, background: bool = True):
        """Pre-encode every KB chunk, in batches of 64.

        First-touch MMR otherwise costs ~8 embed calls (~2-3s CPU) for the
        first visitor who asks about an untouched section.

        ``background=False`` runs inline - used at service boot so the
        whole-KB cost (~60s CPU) lands before the service reports ready.
        Background warming must NOT be mixed with live queries: OMP
        serializes parallel regions process-wide, so a concurrent query
        waits behind the entire batch (measured: 68s first-turn retrieval).
        Sets ``vectors_warm`` either way.
        """
        import threading

        def run():
            started = time.perf_counter()
            try:
                pending = [
                    m
                    for m in self.docs_meta
                    if m["chunk_id"] not in self._mmr_vec_cache
                ]
                for i in range(0, len(pending), 64):
                    batch = pending[i : i + 64]
                    texts = [f"[{m['path']}] {m['text']}" for m in batch]
                    vectors = self.embedder.encode(texts, normalize=True)
                    for meta, vec in zip(batch, vectors, strict=False):
                        self._mmr_vec_cache[meta["chunk_id"]] = vec
                logger.info(
                    "mmr vector cache warmed: %d chunks in %.1fs",
                    len(self._mmr_vec_cache),
                    time.perf_counter() - started,
                )
            except Exception:
                logger.exception("vector cache warm failed")
            finally:
                self._vectors_warm.set()

        if not background:
            run()
            return None
        thread = threading.Thread(target=run, name="mmr-warm", daemon=True)
        thread.start()
        return thread

    @property
    def vectors_warm(self) -> bool:
        """True once every KB chunk vector is cached (or warming failed)."""
        return self._vectors_warm.is_set()

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
    def _effective_query(self, query: str, memory=None) -> str:
        """Enrich very short pronoun follow-ups with recent topics."""
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
        q_vec: np.ndarray | None = None,
    ) -> RetrievalResult:
        """Always search; include docs only when EVIDENCE supports it.

        Replaces the old keyword/centroid gate: instead of predicting
        utterance type from surface forms, we measure whether the KB
        actually contains something similar to what was said. Below the
        bar -> no docs block -> the LLM continues the conversation from
        native history alone.
        """
        started = time.perf_counter()
        result = RetrievalResult(query_used=query)

        if not self.ready:
            logger.warning("retriever called before index load")
            return result

        effective_query = self._effective_query(query, memory)
        result.query_used = effective_query
        if q_vec is None or effective_query != query:
            q_vec = self._embed_query(effective_query)

        k = min(self.cfg.retriever_topk_candidates, self.index.ntotal)  # type: ignore[union-attr]
        scores, ids = self.index.search(q_vec[None, :].astype(np.float32), k)  # type: ignore[union-attr]

        raw_scores = [float(s) for s, i in zip(scores[0], ids[0], strict=False) if i >= 0]
        best_sim = max(raw_scores) if raw_scores else 0.0
        result.best_sim = best_sim
        if best_sim < self.cfg.evidence_sim_min:
            logger.info(
                "evidence %.3f < %.2f - no docs (%s)",
                best_sim,
                self.cfg.evidence_sim_min,
                effective_query[:50],
            )
            result.elapsed_s = time.perf_counter() - started
            return result
        candidates: list[tuple[dict, float]] = []
        seen_turn_window = set()
        if memory is not None:
            seen_turn_window = memory.recently_seen_chunk_ids(
                self.cfg.dedup_window_turns
            )
        penalty_applied: dict[str, float] = {}
        for raw_score, idx in zip(scores[0], ids[0], strict=False):
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
