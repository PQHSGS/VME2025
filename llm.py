"""Pluggable streaming LLM backends.

Backends speak a common protocol:
    stream(messages, **overrides) -> Iterator[str]   (token chunks)
    complete(messages, **overrides) -> str           (non-streaming)
    health_check() -> bool

``gemini`` : Google Gemini (flash-lite tier) via google-genai SDK - the
             production path for this kiosk.
``mock``   : deterministic offline stream for tests/dev/--check.

Local OpenAI-compatible serving was pruned: on the kiosk's CPU/small-GPU
boxes no self-hosted model beat flash-lite on TTFA *and* quality, and the
circuit breaker covers outages more gracefully than a weak fallback model.
If offline generation ever becomes a requirement, reintroduce a backend
behind the same protocol rather than special-casing call sites.

``select_backend`` fails loud at startup so a kiosk never silently degrades
to canned mock replies.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Protocol

logger = logging.getLogger("llm")


class LLMBackend(Protocol):
    name: str

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...
    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...
    def health_check(self) -> bool: ...


# ----------------------------------------------------------------------
class MockBackend:
    """Offline backend: streams a canned reply so the pipeline is testable."""

    name = "mock"

    CANNED = (
        "À câu này hay đó cháu à. Theo tài liệu trong bảo tàng thì "
        "Tết Trung Thu có múa lân, rước đèn và bày mâm ngũ quả lắm nhé. "
        "Cháu muốn ông kể kỹ hơn không?"
    )

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        # emit small word-chunks to exercise the sentence splitter
        words = self.CANNED.split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "")
            time.sleep(0.01)

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "Tóm tắt: trẻ hỏi về Trung Thu, ông giới thiệu múa lân và rước đèn."

    def health_check(self) -> bool:
        return True


# ----------------------------------------------------------------------
class GeminiBackend:
    """Google Gemini streaming via google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_level: str | None = "minimal",
        base_url: str | None = None,
    ):
        from google import genai  # lazy heavy-ish import

        self.name = "gemini"
        self.model = model
        self.thinking_level = thinking_level
        # HTTP/2 + keepalive: one warm connection matters after idle gaps.
        # (httpx ships HTTP/2 disabled; opt in per SDK docs.)
        http_options = {
            "client_args": {"http2": True},
            "async_client_args": {"http2": True},
        }
        if base_url:
            http_options["base_url"] = base_url
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        return "\n\n".join(system_parts), rest

    def _config(
        self, temperature: float | None, max_tokens: int | None, system_instruction: str
    ):
        from google.genai import types

        thinking = None
        if self.thinking_level:
            thinking = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.4,
            max_output_tokens=max_tokens or 220,
            # Per-call value: the summarizer thread calls complete() on this
            # same backend instance - shared mutable state would race stream().
            system_instruction=system_instruction or None,
            thinking_config=thinking,
        )

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        system_instruction, contents = self._split_messages(messages)
        payload = [
            {
                "role": "user" if m["role"] != "assistant" else "model",
                "parts": [{"text": m["content"]}],
            }
            for m in contents
        ]
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=payload,
            config=self._config(temperature, max_tokens, system_instruction),  # type: ignore[arg-type]
        ):
            if chunk.text:
                yield chunk.text

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "".join(self.stream(messages, temperature, max_tokens))

    def health_check(self) -> bool:
        try:
            for _ in self.stream(
                [
                    {"role": "system", "content": "trả lời đúng một chữ: ok"},
                    {"role": "user", "content": "ok"},
                ],
                max_tokens=8,
            ):
                return True
            return False
        except Exception as exc:
            logger.debug("gemini unhealthy: %s", exc)
            return False


# ----------------------------------------------------------------------
def select_backend(cfg) -> LLMBackend:
    """Resolve cfg.llm_backend ("auto"|"gemini"|"mock") to an instance."""
    mode = cfg.llm_backend.lower()
    errors: list[str] = []

    if mode == "mock":
        return MockBackend()

    if mode in ("auto", "gemini"):
        if not cfg.gemini_api_key:
            errors.append("GEMINI_API_KEY is not set")
        else:
            try:
                gemini = GeminiBackend(
                    api_key=cfg.gemini_api_key,
                    model=cfg.gemini_model,
                    thinking_level=cfg.gemini_thinking_level,
                    base_url=cfg.gemini_base_url or None,
                )
                if gemini.health_check():
                    logger.info("LLM backend: gemini (%s)", cfg.gemini_model)
                    return gemini
                errors.append("gemini health check failed")
            except Exception as exc:
                errors.append(f"gemini init failed: {exc}")

    # Fail loud: a kiosk that silently answers every question with the mock
    # canned line is worse than one that refuses to start.
    raise RuntimeError(
        f"no live LLM backend available (mode={mode}): {'; '.join(errors) or 'no backends configured'}"
    )
