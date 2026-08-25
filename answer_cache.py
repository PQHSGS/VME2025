"""Semantic answer cache - replay previous replies for repeated questions.

Kiosk visitors ask the same handful of questions all day. After one full
RAG+LLM turn, a near-identical later question replays the stored reply
instantly: no retrieval, no Gemini round-trip - the single largest possible
TTFA win for this venue. Two tiers (GPTCache pattern, arXiv 2303.06749):

  1. exact match on normalized text (verbatim repeats are common from ASR)
  2. embedding cosine >= threshold over past entries

False positives are the classic failure mode ("còn trẻ em thì sao?" scoring
close to "giá vé?"), so both write and read sides are guarded:
  * entries are only stored for self-contained, doc-grounded answers
    (enforced by the orchestrator: no query enrichment, >=1 retrieved doc,
    no barge-in truncation, minimum reply length);
  * follow-up-shaped queries never hit the cache;
  * the similarity bar is deliberately high and TTL + LRU bound staleness.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict

import numpy as np

logger = logging.getLogger("rag.answer_cache")

# Follow-up fragments that must resolve against conversation context - they
# must never be answered from (or written into) the cache.
_FOLLOWUP_STARTS = (
    "còn ",
    "vậy ",
    "thì sao",
    "nó ",
    "noi ",
    "anh nói",
    "chị nói",
    "ý ông",
    "giống như",
    "tại sao ",
    "làm sao ",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().strip("?.!,")).strip()


def is_cacheable_query(text: str) -> bool:
    """Conservative shape check for cache reads AND writes."""
    q = normalize(text)
    if len(q) < 10 or len(q.split()) < 2:
        return False
    return not q.startswith(_FOLLOWUP_STARTS)


class AnswerCache:
    """Bounded LRU + TTL store of (query vector, reply)."""

    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self._entries: OrderedDict[str, dict] = OrderedDict()  # norm query -> entry
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return (
            bool(self.cfg.answer_cache_enabled)
            and self.embedder is not None
        )

    # ------------------------------------------------------------------
    def lookup(self, query: str, q_vec: np.ndarray | None = None) -> str | None:
        if not self.enabled or not is_cacheable_query(query):
            return None
        key = normalize(query)
        with self._lock:
            self._purge_expired()
            exact = self._entries.get(key)
            if exact is not None:
                self._entries.move_to_end(key)
                logger.info("answer-cache EXACT hit: %r", query[:60])
                return exact["reply"]

            threshold = float(self.cfg.answer_cache_similarity)
            if threshold <= 0 or not self._entries:
                return None
            vec = q_vec if q_vec is not None else self.embedder.encode_query(query)
            best_key, best_sim = None, 0.0
            for ekey, entry in self._entries.items():
                sim = float(entry["vec"] @ vec)
                if sim > best_sim:
                    best_key, best_sim = ekey, sim
            if best_sim >= threshold:
                self._entries.move_to_end(best_key)
                entry = self._entries[best_key]
                logger.info(
                    "answer-cache SEMANTIC hit (%.3f): %r ~ %r",
                    best_sim,
                    query[:40],
                    entry["query"][:40],
                )
                return entry["reply"]
        return None

    def store(self, query: str, reply: str, q_vec: np.ndarray | None = None) -> None:
        if not self.enabled or not is_cacheable_query(query):
            return
        reply = reply.strip()
        if len(reply) < self.cfg.answer_cache_min_reply_chars:
            return
        key = normalize(query)
        vec = q_vec if q_vec is not None else self.embedder.encode_query(query)
        max_entries = self.cfg.answer_cache_max_entries
        ttl_s = self.cfg.answer_cache_ttl_min * 60
        with self._lock:
            self._entries[key] = {
                "query": query,
                "reply": reply,
                "vec": vec,
                "expires": time.monotonic() + ttl_s,
            }
            self._entries.move_to_end(key)
            while len(self._entries) > max_entries:
                self._entries.popitem(last=False)
        logger.debug("answer-cache stored %r (%d chars)", query[:60], len(reply))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, e in self._entries.items() if e["expires"] <= now]
        for k in expired:
            self._entries.pop(k, None)

