"""Incremental sentence segmentation for streaming LLM output.

Tokens arrive one fragment at a time; we yield complete sentences as soon as
they end so TTS can start synthesizing while the LLM keeps generating.
Handles Vietnamese text: basic punctuation, decimals, common abbreviations,
and force-splits overly long runs so TTS chunks stay small and responsive.

TTFA trick (first-chunk clause emission): Vietnamese replies typically open
with a short discourse marker ("Đúng,", "Dạ, ông có thể..."). When enabled,
the very first chunk may be released at the first comma instead of waiting
for terminal punctuation - VieNeu starts speaking ~a full clause earlier.
Only applies once, before any real sentence has been emitted.
"""

from __future__ import annotations

import re

# Sentence-ending punctuation followed by whitespace or end of stream.
_SENT_END = re.compile(r"[.!?…]+[\s]*")
# Decimal numbers like 1.5 - never split inside.
_DECIMAL = re.compile(r"\d[.,]\d")
# Trailing word of a candidate - abbreviation check.
_TAIL_WORD = re.compile(r"([A-Za-zÀ-ỹĐđ\.]+)$")
# Markdown artifacts the TTS should not read aloud.
_MARKDOWN = re.compile(r"[*_`#>]+")
# First-clause cut: ", " after a plausible opening marker.
_FIRST_COMMA = re.compile(r", ")
_ABBREV = (
    "tp",
    "ts",
    "bs",
    "ths",
    "pgs",
    "gs",
    "vv",
    "v.v",
    "etc",
    "dr",
    "mr",
    "mrs",
    "hn",
    "hcm",
    "đh",
    "thpt",
    "thcs",
)
_MAX_SENT_CHARS = 240  # force-split longer than this at a soft boundary
_MIN_SENT_CHARS = 12  # merge tiny fragments with the next sentence


class SentenceSplitter:
    """Push text fragments in, get complete sentences out."""

    def __init__(
        self,
        min_chars: int = _MIN_SENT_CHARS,
        max_chars: int = _MAX_SENT_CHARS,
        early_first_clause: bool = False,
    ):
        self._buf = ""
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.early_first_clause = early_first_clause
        self._emitted_any = False

    @staticmethod
    def clean(text: str) -> str:
        """Strip markdown artifacts the TTS should not read aloud."""
        return _MARKDOWN.sub(" ", text)

    def push(self, chunk: str) -> list[str]:
        self._buf += self.clean(chunk)
        out: list[str] = []
        while True:
            sentence, rest = self._pop_sentence(self._buf)
            if sentence is None:
                break
            out.append(sentence)
            self._buf = rest
        if out:
            self._emitted_any = True
        elif self.early_first_clause and not self._emitted_any:
            clause = self._pop_first_clause(self._buf)
            if clause:
                out.append(clause)
                self._emitted_any = True
        # Force-split pathological runs (no punctuation for a long time).
        while len(self._buf) > self.max_chars:
            cut = self._soft_cut_index(self._buf, self.max_chars)
            out.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:].lstrip()
            self._emitted_any = True
            if not self._buf:
                break
        return [s for s in (part.strip() for part in out) if s]

    def flush(self) -> list[str]:
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []

    # ------------------------------------------------------------------
    def _pop_first_clause(self, buf: str) -> str | None:
        """Release 'Vâng,' / 'Dạ, ông có thể...' style openings immediately.

        Bounded window so we never emit a fragment that is either too tiny to
        synthesize cleanly or so long it defeats the purpose.
        """
        m = _FIRST_COMMA.search(buf[:80])
        if not m:
            return None
        clause = buf[: m.start()].strip()
        if len(clause) < 3 or len(clause) > 60:
            return None
        self._buf = buf[m.end() :]
        return clause

    def _pop_sentence(self, buf: str) -> tuple[str | None, str]:
        """Find the next real sentence end, skipping false boundaries
        (decimals, abbreviations, too-short fragments) by advancing."""
        pos = 0
        while True:
            match = _SENT_END.search(buf, pos)
            if not match:
                return None, buf
            end = match.end()
            candidate = buf[:end]
            # Keep decimals together ("1.5 triệu") - resume search after dot.
            window = candidate[-3:] + (buf[end : end + 2] or "")
            if _DECIMAL.search(window):
                pos = max(end, match.start() + 1)
                continue
            # Keep abbreviations ("TP. Hồ Chí Minh", "v.v...").
            tail_word = _TAIL_WORD.search(candidate.rstrip())
            if tail_word and tail_word.group(1).lower().rstrip(".") in _ABBREV:
                pos = end
                continue
            if len(candidate.strip()) < self.min_chars:
                # merge tiny leading fragments with the next sentence
                pos = end
                continue
            return candidate.strip(), buf[end:]

    @staticmethod
    def _soft_cut_index(buf: str, limit: int) -> int:
        window = buf[:limit]
        for sep in (", ", "; ", " và ", " rồi ", " "):
            idx = window.rfind(sep)
            if idx > limit // 2:
                return idx + len(sep)
        return limit
