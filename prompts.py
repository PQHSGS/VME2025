"""Prompt assembly - where context becomes a strict, budgeted structure.

Layout sent to the LLM each turn:

    [system]    persona + rules + how to treat context blocks
    [user]*     native history turns (verbatim, alternating roles)
    [user]      === BỐI CẢNH ===            (only when there is any)
                   TÓM TẮT: rolling summary
                   THÔNG TIN ĐÃ BIẾT: facts
                === TÀI LIỆU THAM KHẢO ===  (only when retrieval fired)
                   numbered chunks with breadcrumb paths
                === GỢI Ý TRẢ LỜI ===       (operator steering, optional)
                === CÂU HỎI HIỆN TẠI ===    raw user message

History as REAL role turns lets the provider enforce continuity; everything
disposable (docs/summary/facts) rides only in the final turn so it never
compounds into memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("prompts")

DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.md"
)


def load_system_prompt(path: Path | None = None) -> str:
    path = path or DEFAULT_SYSTEM_PROMPT_PATH
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.exception(
            "cannot read system prompt at %s; using minimal fallback", path
        )
        return (
            "Bạn là Ông Tiến sĩ Giấy AI, trợ lý vui tính cho trẻ em tại "
            "Bảo tàng Dân tộc học Việt Nam. Trả lời ngắn gọn bằng tiếng Việt."
        )


def format_retrieved_block(docs: list[dict]) -> str:
    """docs: [{'path': str, 'text': str, 'score': float}] -> numbered text."""
    lines: list[str] = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] {doc['path']}\n{doc['text'].strip()}")
    return "\n\n".join(lines)


def build_context_block(
    memory,
    docs: list[dict] | None,
    guidance: str | None = None,
) -> str | None:
    """Assemble the non-verbatim layers: summary + facts + docs + guidance."""
    sections: list[str] = []
    if memory.summary:
        sections.append(f"### TÓM TẮT CÁC PHẦN TRƯỚC\n{memory.summary.strip()}")
    if memory.facts:
        facts = "; ".join(f"{k}: {v}" for k, v in memory.facts.items())
        sections.append(f"### THÔNG TIN ĐÃ BIẾT VỀ EM NHÍ\n{facts}")
    if docs:
        sections.append(
            "### TÀI LIỆU NỀN THAM KHẢO (chỉ dùng khi em nhí hỏi để tìm hiểu"
            " sâu hơn; đang trò chuyện thường thì bỏ qua)\n"
            + format_retrieved_block(docs)
        )
    # Operator-scripted steering from a situations.csv row that had no
    # canned answer - advice on HOW to answer, not what to say verbatim.
    if guidance:
        sections.append(f"### GỢI Ý TRẢ LỜI (từ ban tổ chức)\n{guidance.strip()}")
    if not sections:
        return None
    return "\n\n".join(sections)


def build_messages(
    system_prompt: str,
    memory,
    user_text: str,
    docs: list[dict] | None = None,
    history_limit: int | None = None,
    guidance: str | None = None,
) -> tuple[list[dict], dict]:
    """[system] + NATIVE alternating history turns + [final structured payload].

    History rides as real user/assistant turns so provider APIs enforce role
    continuity structurally - the model cannot treat a mid-conversation turn
    as a fresh Q&A (this killed the re-greeting/tone-drift class of bugs).
    Summary/facts/docs/guidance stay in the FINAL turn only: disposable
    per-turn context that never compounds into history.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    recent = list(memory.recent)[-(history_limit or len(memory.recent)) :]
    for exchange in recent:
        if exchange.user:
            messages.append({"role": "user", "content": exchange.user})
        if exchange.bot:
            messages.append({"role": "assistant", "content": exchange.bot})

    context_block = build_context_block(memory, docs, guidance)
    current_parts: list[str] = []
    if context_block:
        current_parts.append(context_block)
    current_parts.append(f"### CÂU HỎI HIỆN TẠI CỦA EM NHÍ\n{user_text}")
    messages.append({"role": "user", "content": "\n\n---\n\n".join(current_parts)})

    meta = {
        "context_chars": len(context_block or ""),
        "recent_turns": len(recent),
        "docs": len(docs or []),
        # Total characters shipped to the LLM - the input-side TTFT lever.
        "prompt_chars": len(system_prompt)
        + sum(len(str(m.get("content", ""))) for m in messages[1:]),
    }
    return messages, meta
