"""Scripted-question fast path (from data/situations.csv).

Frequent operator-scripted questions (greetings, "who are you?", "how old
are you?") are answered instantly from curated answers - no LLM call,
sub-millisecond decision after embedding. Falls back silently when the
embedding model is unavailable.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass

import numpy as np

from answer_cache import normalize as _normalize  # single normalization source
from config import SITUATIONS_CSV

logger = logging.getLogger("rag.situations")


@dataclass
class Situation:
    question: str
    guidance: str
    answer: str
    score: float


class SituationMatcher:
    def __init__(self, cfg, embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.rows: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._exact: dict[str, dict] = {}

    def load(self) -> bool:
        path = SITUATIONS_CSV
        try:
            with open(path, newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    question = (row.get("Câu hỏi") or "").strip()
                    if not question:
                        continue
                    entry = {
                        "question": question,
                        "guidance": (row.get("Hướng dẫn") or "").strip(),
                        "answer": (row.get("Câu trả lời mẫu") or "").strip(),
                    }
                    self.rows.append(entry)
                    self._exact[_normalize(question)] = entry
        except OSError as exc:
            logger.error("cannot read situations csv %s: %s", path, exc)
            return False
        if self.rows and not self.cfg.situations_enabled:
            logger.info(
                "situations disabled by config; rows loaded anyway for reference"
            )
        if self.rows:
            self.vectors = self.embedder.encode(
                [r["question"] for r in self.rows], normalize=True
            )
            logger.info("situations loaded: %d rows", len(self.rows))
        return bool(self.rows)

    @property
    def ready(self) -> bool:
        return bool(self.rows) and self.cfg.situations_enabled

    def match(self, query: str, q_vec: np.ndarray | None = None) -> Situation | None:
        if not self.ready:
            return None
        normalized = _normalize(query)
        exact = self._exact.get(normalized)
        if exact:
            return Situation(score=1.0, **exact)
        assert self.vectors is not None
        vec = q_vec if q_vec is not None else self.embedder.encode_query(query)
        sims = self.vectors @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= self.cfg.situations_threshold:
            row = self.rows[best]
            logger.info("situation matched (%.3f): %r", score, row["question"])
            return Situation(score=score, **row)
        return None
