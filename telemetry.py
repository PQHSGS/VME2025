"""Per-turn telemetry and conversation transcript.

Two output files per session in ``logs/``:

``traces.jsonl``
    Latency-relevant stage timeline so offline analysis can answer
    "where did the second go?" without touching live code paths again.
    Marks in use (first occurrence wins, relative to turn start):

        turn_start -> situation | answer_cache -> retrieval -> llm_ttft

``conversations.jsonl``
    Full conversation transcript: every user utterance and bot reply with
    timing, retrieval path, and token counts. One JSON object per turn.
    Use for post-show analysis, golden-set creation, and visitor Q&A review.

Everything is best-effort: tracing failures are swallowed and never
propagate into the conversation loop.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("telemetry")


class NullSpan:
    """No-op span used when telemetry is disabled or fails."""

    _extra: dict = {}

    def mark(self, stage: str) -> None: ...
    def set(self, **fields) -> None: ...
    def finish(self, **fields) -> None: ...


class TurnSpan(NullSpan):
    def __init__(self, path: Path | None, session_id: str, user_text: str):
        self._path = path
        self._session_id = session_id
        self._user = user_text[:120]
        self._t0 = time.perf_counter()
        self._marks: dict[str, float] = {"turn_start": 0.0}
        self._extra: dict = {}

    def mark(self, stage: str) -> None:
        if stage not in self._marks:
            self._marks[stage] = round(time.perf_counter() - self._t0, 3)

    def set(self, **fields) -> None:
        self._extra.update(fields)

    def finish(self, **fields) -> None:
        self.set(**fields)
        if self._path is None:
            return
        try:
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session": self._session_id,
                "user": self._user,
                "stages_ms": {
                    k: int(v * 1000)
                    for k, v in sorted(self._marks.items(), key=lambda kv: kv[1])
                },
                **self._extra,
            }
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("trace write failed", exc_info=True)


class Tracer:
    def __init__(self, enabled: bool, log_dir: Path):
        self.path = (log_dir / "traces.jsonl") if enabled else None

    def start(self, session_id: str, user_text: str) -> TurnSpan | NullSpan:
        if self.path is None:
            return NullSpan()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return TurnSpan(self.path, session_id, user_text)
        except Exception:
            logger.debug("tracer init failed", exc_info=True)
            return NullSpan()


class ConversationLogger:
    """Writes a clean turn-by-turn transcript to logs/conversations.jsonl.

    Unlike traces.jsonl (latency diagnostics), this file records full user
    text and bot reply for post-show analysis, golden-set creation, and
    visitor Q&A review.  One JSON object per turn.
    """

    def __init__(self, enabled: bool, log_dir: Path):
        self._path = (log_dir / "conversations.jsonl") if enabled else None

    def log_turn(
        self,
        session_id: str,
        user_text: str,
        reply: str,
        path: str,
        elapsed_s: float,
        ttft_s: float | None = None,
        docs: int = 0,
    ) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session": session_id,
                "user": user_text,
                "reply": reply,
                "path": path,
                "elapsed_s": round(elapsed_s, 3),
                "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
                "docs": docs,
                "reply_chars": len(reply),
            }
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("conversation log write failed", exc_info=True)
