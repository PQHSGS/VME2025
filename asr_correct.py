"""Deterministic domain-homophone correction for ASR transcripts.

gipformer's remaining errors on kid speech are mostly homophones of the
kiosk's proper nouns ("chú quậy" for "chú Cuội", "múa lan" for "múa lân").
A full seq2seq corrector (e.g. ProtonX legal-tc) is the wrong tool here:
legal/OCR-domain, 128-token input, beam-search latency on the TTFA hot
path. This module is the cheap 90% solution - exact phrase mapping applied
post-transcribe in microseconds.

Extend at the venue by appending rows to data/asr_homophones.csv
(columns: sai, dung) - no code change, picked up on next restart.
"""

from __future__ import annotations

import csv
import logging
import re

from config import BASE_DIR

logger = logging.getLogger("asr.correct")

# Near-homophone -> canonical domain term (lowercase keys; longest match wins).
DEFAULT_MAP = {
    "chú quậy": "chú Cuội",
    "chú củi": "chú Cuội",
    "chú kười": "chú Cuội",
    "chị hành": "chị Hằng",
    "chị hàng": "chị Hằng",
    "chú hằng": "chị Hằng",
    "thổ ngọc": "thỏ ngọc",
    "tỏ ngọc": "thỏ ngọc",
    "đèn ông xao": "đèn ông sao",
    "múa lan": "múa lân",
    "lân sư rông": "lân sư rồng",
    "sư rông": "sư rồng",
    "dối nước": "rối nước",
    "trống đông": "trống đồng",
    "tô he": "tò he",
    "phổng đất": "phỗng đất",
    "tiền sĩ giấy": "tiến sĩ giấy",
    "tiếng sĩ giấy": "tiến sĩ giấy",
    "bảo tang": "bảo tàng",
    "ảo tàng": "bảo tàng",
    "cánh chiếu": "cánh diều",
}

_CSV_PATH = BASE_DIR / "data" / "asr_homophones.csv"


def load_map() -> dict[str, str]:
    """Defaults + optional venue CSV overrides (merged, CSV wins)."""
    mapping = dict(DEFAULT_MAP)
    try:
        with open(_CSV_PATH, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                wrong = (row.get("sai") or "").strip().lower()
                right = (row.get("dung") or "").strip()
                if wrong and right:
                    mapping[wrong] = right
        if len(mapping) != len(DEFAULT_MAP):
            logger.info(
                "asr homophones: %d venue overrides from %s",
                len(mapping) - len(DEFAULT_MAP),
                _CSV_PATH,
            )
    except OSError:
        pass  # no venue file - defaults only
    return mapping


_MAP = None
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)(\w)", re.UNICODE)


def correct_transcript(text: str) -> str:
    """Replace known near-homophones with canonical terms. Longest-first so
    'lân sư rông' wins over its substring 'sư rông'."""
    global _MAP
    if _MAP is None:
        _MAP = load_map()
    if not text or not _MAP:
        return text
    result = text.lower()
    changed = []
    for wrong in sorted(_MAP, key=len, reverse=True):
        if wrong in result:
            result = result.replace(wrong, _MAP[wrong])
            changed.append(f"{wrong}->{_MAP[wrong]}")
    if changed:
        logger.info("asr correct: %s | %r", "; ".join(changed), text[:60])
        # Re-capitalize sentence starts lost by lowercasing.
        result = _SENTENCE_START.sub(
            lambda m: f"{m.group(1)}{m.group(2).upper()}", result
        )
        return result.strip() or text
    return text
