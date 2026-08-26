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
        tools: bool = False,
        memory_ctx: dict | None = None,
    ) -> Iterator[str]: ...
    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...
    def health_check(self) -> bool: ...


# Tool contract shared by every backend/consumer. After stream() finishes,
# ``last_tool_events`` lists one entry per executed search and
# ``tool_skipped`` says whether the model answered without searching.
KB_TOOL_NAME = "search_kb"


def build_kb_tool_declaration():
    """Typed Gemini tool declaration for search_kb (import-time cheap)."""
    from google.genai import types

    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=KB_TOOL_NAME,
                description=(
                    "Tra cứu tài liệu nội bộ về Bảo tàng Dân tộc học và Tết "
                    "Trung Thu. VỚI MỌI câu hỏi liên quan đến đèn ông sao, múa "
                    "lân, bánh Trung Thu, chú Cuội, chị Hằng, rối nước, trò "
                    "chơi dân gian hay bảo tàng: LUÔN gọi trước khi trả lời - "
                    "kể cả khi em nhí chỉ gật đầu ('có ạ', 'tiếp đi'). Chỉ "
                    "trả lời trực tiếp cho chào hỏi / cảm ơn / trò chuyện "
                    "thuần túy. Query phải ĐỦ NGHĨA dựa trên toàn bộ hội thoại."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Cụm từ khóa tìm kiếm, tự chứa đủ ngữ nghĩa.",
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )


# ----------------------------------------------------------------------
class MockBackend:
    """Offline backend: streams a canned reply so the pipeline is testable.

    Tool-mode simulation: set ``tool_calls_first`` to make the backend
    invoke the executor before answering; ``tool_skips`` to answer without
    searching. Either way ``last_tool_events``/``tool_skipped`` mirror the
    GeminiBackend contract so orchestrator code paths are identical.
    """

    name = "mock"
    tool_calls_first = False
    tool_skips = False
    search_query = "đèn ông sao làm bằng gì"

    CANNED = (
        "À câu này hay đó cháu à. Theo tài liệu trong bảo tàng thì "
        "Tết Trung Thu có múa lân, rước đèn và bày mâm ngũ quả lắm nhé. "
        "Cháu muốn ông kể kỹ hơn không?"
    )

    def __init__(self):
        self.last_tool_events: list[dict] = []
        self.tool_skipped = True
        self.last_stream_kwargs: dict = {}

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: bool = False,
        memory_ctx: dict | None = None,
        tool_executor=None,
        force_search: bool = True,
    ) -> Iterator[str]:
        self.last_stream_kwargs = {
            "tools": tools,
            "memory_ctx": memory_ctx,
            "force_search": force_search,
        }
        self.last_tool_events = []
        fired = tools and not self.tool_skips
        if fired:
            docs_text = (
                tool_executor(self.search_query) if tool_executor is not None else ""
            )
            self.last_tool_events = [
                {"query": self.search_query, "docs": 1 if docs_text else 0,
                 "best_sim": 0.7}
            ]
        self.tool_skipped = not bool(self.last_tool_events)
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
        from google import genai
        from google.genai import types

        self.name = "gemini"
        self.model = model
        self.thinking_level = thinking_level
        self.last_tool_events: list[dict] = []
        self.tool_skipped = True
        # HTTP/2 + keepalive: one warm connection matters after idle gaps.
        # Must be a TYPED HttpOptions: google-genai 2.x mishandles base_url
        # when handed the legacy raw dict (request URL loses its scheme).
        options = types.HttpOptions(
            client_args={"http2": True},
            async_client_args={"http2": True},
        )
        if base_url:
            options.base_url = base_url
        self._client = genai.Client(api_key=api_key, http_options=options)

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        # google-genai 2.x rejects requests whose LAST content turn is a
        # model turn ("Requests ending with a model turn are not supported").
        # Our prompt contract ends with the pre-ack assistant turn
        # (prompts.build_messages) - valid for OpenAI-style APIs, illegal
        # here, so strip trailing model turns at this boundary.
        while rest and rest[-1].get("role") == "assistant":
            rest.pop()
        return "\n\n".join(system_parts), rest

    def _config(
        self,
        temperature: float | None,
        max_tokens: int | None,
        system_instruction: str,
        tools: bool = False,
        force_search: bool = True,
    ):
        from google.genai import types

        thinking = None
        if self.thinking_level:
            thinking = types.ThinkingConfig(thinking_level=self.thinking_level)
        tool_cfg = None
        if tools:
            tool_cfg = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    # ANY = the model must call search_kb every turn; the
                    # executor answers chit-chat searches with an empty
                    # result, and the model's own judgment picks the query.
                    mode="ANY" if force_search else "AUTO"
                )
            )
        return types.GenerateContentConfig(
            temperature=temperature if temperature is not None else 0.4,
            max_output_tokens=max_tokens or 220,
            # Per-call value: the summarizer thread calls complete() on this
            # same backend instance - shared mutable state would race stream().
            system_instruction=system_instruction or None,
            thinking_config=thinking,
            tools=[build_kb_tool_declaration()] if tools else None,
            tool_config=tool_cfg,
            # Streaming + automatic function calling don't compose in the
            # SDK; we drive the two-leg loop ourselves below.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
            if tools
            else None,
        )

    def _consume_stream(self, chunks, sink: list, state: dict):
        """Yield text tokens from one streaming leg, collecting typed parts.

        Parts (including ``thought_signature`` carriers) are appended to
        ``sink`` so a follow-up tool leg can replay them verbatim; the last
        raw chunk lands in ``state["last"]`` for usage logging.
        """
        for chunk in chunks:
            state["last"] = chunk
            try:
                candidate = chunk.candidates[0]
                content = candidate.content
                if content and content.parts:
                    sink.extend(content.parts)
                    for part in content.parts:
                        if part.text:
                            yield part.text
            except AttributeError:
                pass

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: bool = False,
        memory_ctx: dict | None = None,
        tool_executor=None,
        force_search: bool = True,
    ) -> Iterator[str]:
        system_instruction, contents = self._split_messages(messages)
        payload = [
            {
                "role": "user" if m["role"] != "assistant" else "model",
                "parts": [{"text": m["content"]}],
            }
            for m in contents
        ]
        state: dict = {"last": None}
        self.last_tool_events: list[dict] = []
        self.tool_skipped = True
        decision_config = self._config(
            temperature, max_tokens, system_instruction, tools, force_search
        )
        try:
            # Phase 1 - decision leg (optionally forced via ANY mode).
            parts: list[dict] = []
            for token in self._consume_stream(
                self._client.models.generate_content_stream(
                    model=self.model, contents=payload, config=decision_config
                ),
                parts,
                state,
            ):
                yield token

            fc_parts = [p for p in parts if p.function_call is not None]
            if not (tools and tool_executor is not None and fc_parts):
                return  # answered directly (or nothing to execute)

            # Phase 2 - execute every requested search...
            responses: list[dict] = []
            for fc in (p.function_call for p in fc_parts):
                query = str((fc.args or {}).get("query", ""))[:200]
                docs_text = tool_executor(query)
                if not docs_text:
                    # Empty results must STEER, not invite another call.
                    docs_text = (
                        "KHÔNG tìm thấy tài liệu phù hợp. Hãy trả lời em nhí "
                        "bằng hiểu biết chung của Ông một cách thân thiện, và "
                        "không bịa chi tiết về bảo tàng."
                    )
                self.last_tool_events.append({"query": query})
                self.tool_skipped = False
                responses.append(
                    {
                        "function_response": {
                            "name": fc.name or KB_TOOL_NAME,
                            "response": {"result": docs_text},
                        }
                    }
                )
            contents_running = payload + [
                {"role": "model", "parts": parts},
                {"role": "user", "parts": responses},
            ]

            # Phase 3 - FINAL answer leg with tools STRIPPED. Flash-lite
            # sometimes re-invokes the tool when tools remain visible;
            # removing them makes a text answer structurally guaranteed.
            final_config = self._config(
                temperature, max_tokens, system_instruction, tools=False
            )
            yield from self._consume_stream(
                self._client.models.generate_content_stream(
                    model=self.model,
                    contents=contents_running,
                    config=final_config,
                ),
                [],
                state,
            )
        finally:
            # Token + cache-hit visibility: if cached stays 0 across turns
            # the prompt is below the model's implicit-cache minimum and a
            # decision about explicit caching can be made from real data.
            usage = getattr(state.get("last"), "usage_metadata", None)
            if usage is not None:
                logger.info(
                    "llm tokens: total=%s input=%s output=%s cached=%s",
                    usage.total_token_count,
                    usage.prompt_token_count,
                    usage.candidates_token_count,
                    usage.cached_content_token_count,
                )

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
