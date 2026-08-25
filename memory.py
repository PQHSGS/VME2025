"""Layered conversation memory - the core of context management.

Three layers per session:
  1. ``facts``   - durable attributes extracted with zero-cost regexes
                   (kid's name, likes...). Survive the whole session.
  2. ``summary`` - rolling summary of everything older than the verbatim
                   window. Updated asynchronously every N turns so the
                   summarizer never blocks the realtime path.
  3. ``recent``  - last K exchanges kept verbatim.

Design rule: retrieved documents NEVER enter this structure. They are
assembled fresh each turn by the prompt builder, so history stays clean.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("memory")

_NAME_PATTERNS = [
    re.compile(
        r"\b(?i:tôi|em|cháu|mình|con)\s+tên\s+là\s+([A-ZÀ-ỸĐ][a-zà-ỹđ]+(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ]+){0,2})"
    ),
    re.compile(
        r"\b(?:là|gọi\s+(?i:tôi|em|cháu|con|mình)\s+là)\s+([A-ZÀ-ỸĐ][a-zà-ỹđ]+)\s*(?:nhé|nha|ạ|$)"
    ),
]
_LIKE_PATTERNS = [
    re.compile(
        r"\b(?i:tôi|em|cháu|con|mình)\s+(?:thích|rất thích|yêu|thích nhất|mê)\s+([^.,!?\n]{2,60})",
        re.IGNORECASE,
    ),
]
_PRONOUN_HINTS = ("nó", "cái đó", "cái ấy", "ông ấy", "bà ấy", "vậy", "rồi", "tiếp")


@dataclass
class Exchange:
    user: str
    bot: str
    turn: int
    at: float = field(default_factory=time.time)


class SessionMemory:
    def __init__(
        self, session_id: str, recent_exchanges: int = 4, summary_max_chars: int = 700
    ):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.facts: dict[str, str] = {}
        self.summary: str = ""
        self.recent: deque[Exchange] = deque(maxlen=recent_exchanges)
        self.summary_max_chars = summary_max_chars
        self.turn_count = 0
        self.turns_since_summary = 0
        # chunk_id -> turn number when it was last injected into a prompt
        self.seen_chunks: dict[str, int] = {}
        self._pending_user: str = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def touch(self) -> None:
        self.last_used = time.time()

    def add_user(self, text: str) -> None:
        """Record the raw user message and extract cheap facts from it."""
        with self._lock:
            self.turn_count += 1
            self.turns_since_summary += 1
            self._extract_facts(text)

    def add_bot_reply(self, text: str) -> None:
        with self._lock:
            exchange = Exchange(user=self._pending_user, bot=text, turn=self.turn_count)
            self._pending_user = ""
            self.recent.append(exchange)
            self.touch()

    def amend_last_bot_reply(self, text: str) -> bool:
        """Replace the newest stored reply (barge-in truncation).

        Keeps conversation history faithful to what the child actually heard
        instead of the full generated text. Returns True when amended.
        """
        with self._lock:
            if not self.recent:
                return False
            last = self.recent[-1]
            self.recent[-1] = Exchange(
                user=last.user, bot=text, turn=last.turn, at=last.at
            )
        return True

    def _extract_facts(self, text: str) -> None:
        for pattern in _NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                self.facts.setdefault("tên", match.group(1).strip())
                break
        for pattern in _LIKE_PATTERNS:
            match = pattern.search(text)
            if match:
                like = match.group(1).strip()
                existing = self.facts.get("thích")
                if existing and like.lower() not in existing.lower():
                    self.facts["thích"] = f"{existing}; {like}"
                else:
                    self.facts.setdefault("thích", like)
                break
        self._pending_user = text

    def mark_chunks_shown(self, chunk_ids: list[str]) -> None:
        with self._lock:
            for chunk_id in chunk_ids:
                self.seen_chunks[chunk_id] = self.turn_count

    def recently_seen_chunk_ids(self, window_turns: int = 3) -> set[str]:
        cutoff = self.turn_count - max(0, window_turns - 1)
        return {cid for cid, turn in self.seen_chunks.items() if turn >= cutoff}

    def needs_summary(self, summarize_every_turns: int) -> bool:
        return (
            self.turns_since_summary >= summarize_every_turns and len(self.recent) >= 2
        )

    def overflow_exchanges(self, keep: int = 1) -> list[Exchange]:
        """Exchanges old enough to be folded into the summary."""
        with self._lock:
            return list(self.recent)[: len(self.recent) - keep]

    def apply_summary(self, new_summary: str, summarized_up_to_turn: int) -> None:
        """Replace the rolling summary with the summarizer's consolidated text.

        The background worker already folds the previous summary into its
        prompt, so the result IS the new summary - appending it here would
        duplicate everything and grow context until the cap.
        """
        with self._lock:
            self.summary = new_summary.strip()[: self.summary_max_chars]
            self.turns_since_summary = 0
            while self.recent and self.recent[0].turn <= summarized_up_to_turn:
                self.recent.popleft()

    # ------------------------------------------------------------------
    def is_stale(self, ttl_minutes: float) -> bool:
        return (time.time() - self.last_used) > ttl_minutes * 60

    def looks_like_followup(self, query: str) -> bool:
        """Cheap heuristic for conversational coref ('nó ở đâu vậy?')."""
        stripped = query.strip().strip(".,!?").lower()
        return len(stripped.split()) <= 5 and any(h in stripped for h in _PRONOUN_HINTS)

    def last_topics(self, n_exchanges: int = 2) -> str:
        """Concatenated recent user messages - used to enrich short queries."""
        parts = [ex.user for ex in list(self.recent)[-n_exchanges:] if ex.user]
        return " ".join(parts)


class MemoryManager:
    """Owns sessions; triggers async summaries via an injected summarizer."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sessions: dict[str, SessionMemory] = {}

    def idle_seconds(self, session_id: str) -> float | None:
        """Seconds since the session was last used, WITHOUT touching it."""
        memory = self.sessions.get(session_id)
        if memory is None:
            return None
        return time.time() - memory.last_used

    def get(self, session_id: str) -> SessionMemory:
        memory = self.sessions.get(session_id)
        if memory is None or memory.is_stale(self.cfg.session_ttl_minutes):
            memory = SessionMemory(
                session_id,
                recent_exchanges=self.cfg.recent_exchanges,
                summary_max_chars=self.cfg.summary_max_chars,
            )
            self.sessions[session_id] = memory
            logger.info("new session %s", session_id)
        memory.touch()
        return memory

    def cleanup(self) -> int:
        stale = [
            sid
            for sid, m in self.sessions.items()
            if m.is_stale(self.cfg.session_ttl_minutes)
        ]
        for sid in stale:
            del self.sessions[sid]
        if stale:
            logger.info("evicted %d stale sessions", len(stale))
        return len(stale)

    # ------------------------------------------------------------------
    def maybe_summarize_async(
        self,
        memory: SessionMemory,
        complete_fn: Callable[..., str],
        stop_check: Callable[[], bool] | None = None,
    ) -> threading.Thread | None:
        """Fire a background thread that folds old exchanges into the summary.

        ``complete_fn`` is the LLM non-streaming call (prompt, **kwargs) -> str.
        Returns the thread so callers may join it on shutdown; never blocks
        the realtime loop.
        """
        if not memory.needs_summary(self.cfg.summarize_every_turns):
            return None
        exchanges = memory.overflow_exchanges(keep=1)
        if not exchanges:
            return None
        transcript = "\n".join(
            f"Trẻ: {e.user}\nÔng: {e.bot}" for e in exchanges if e.user or e.bot
        )
        previous = memory.summary
        up_to_turn = exchanges[-1].turn

        def worker() -> None:
            try:
                prompt = (
                    "Tóm tắt ngắn gọn đoạn hội thoại sau giữa một trẻ em và 'Ông Tiến sĩ Giấy' "
                    "(một nhân vật AI tại Bảo tàng Dân tộc học). Giữ lại: tên của trẻ (nếu có), "
                    "sở thích, các chủ đề đã nói, câu hỏi còn chưa trả lời xong. "
                    "Viết dưới 6 dòng tiếng Việt, không thêm bình luận.\n\n"
                    f"TÓM TẮT CŨ (nếu có):\n{previous}\n\nHỘI THOẠI MỚI:\n{transcript}"
                )
                result = complete_fn(prompt, max_tokens=180, temperature=0.2)
                if stop_check and stop_check():
                    return
                if result:
                    memory.apply_summary(result.strip(), up_to_turn)
                    logger.debug("summary updated until turn %d", up_to_turn)
            except Exception:
                logger.exception("background summarization failed")

        thread = threading.Thread(
            target=worker, name=f"summarize-{memory.session_id}", daemon=True
        )
        thread.start()
        return thread
