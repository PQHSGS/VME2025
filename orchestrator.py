"""Conversation orchestrator - the realtime state machine.

Turn flow:
  IDLE -> RECORDING (push-to-talk: ENTER starts, ENTER again sends)
        -> TRANSCRIBING (gipformer via ASR service, ~3s soft budget;
                         homophones canonicalized before the brain sees them)
        -> THINKING     (situation fast-path OR evidence-bar retrieval -> single LLM call;
                         TTFT filler spoken if the model stalls; hard deadline)
        -> SPEAKING     (sentence-level pipelined TTS, barge-in via ENTER)

Everything is injectable so tests/dev mode can run the exact same code path
with mock ASR/LLM/TTS.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from datetime import datetime

from prompts import build_messages, load_system_prompt
from resilience import Deadline, FailureTracker, budget
from sentences import SentenceSplitter
from config import LOG_DIR, SAMPLE_RATE

logger = logging.getLogger("orchestrator")


def _next_attract_line(lines: list[str], idx: int) -> tuple[str, int]:
    """Rotate through invitation lines; returns (line, next_idx)."""
    if not lines:
        return "", idx
    return lines[idx % len(lines)], idx + 1


class ConversationOrchestrator:
    def __init__(
        self,
        cfg,
        *,
        llm=None,
        retriever=None,
        situations=None,
        memory_manager=None,
        tts=None,
        stt=None,
        answer_cache=None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.retriever = retriever
        self.situations = situations
        self.memory_manager = memory_manager
        self.tts = tts
        self.stt = stt
        self.answer_cache = answer_cache  # AnswerCache | None (disabled when None)
        self.system_prompt = load_system_prompt(
            tool_mode=cfg.retrieval_mode in ("auto", "grounded")
        )
        self.session_id = self._new_session_id()
        self.llm_failures = FailureTracker("llm", threshold=3)
        self._llm_cooldown_until = 0.0  # circuit breaker: monotonic deadline
        self._turn_truncated = False  # barge-in/deadline cut the reply short
        self._parked_docs = {"sim": 0.0}  # last turn's best evidence (tool mode)
        self._turn_tool_events: list[dict] = []  # rich per-turn search events
        self._summarizer_threads: list[threading.Thread] = []
        self._barge_in = threading.Event()
        from telemetry import Tracer, ConversationLogger

        self.tracer = Tracer(
            enabled=cfg.telemetry_enabled,
            log_dir=LOG_DIR,
        )
        self.conv_log = ConversationLogger(
            enabled=cfg.telemetry_enabled,
            log_dir=LOG_DIR,
        )

    @staticmethod
    def _new_session_id() -> str:
        return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Load everything that can be loaded before the first visitor talks."""
        started = time.perf_counter()
        if self.retriever is not None:
            self.retriever.load()
            # Only the local retriever warms here; in microservice mode the
            # RAG service blocks on its own whole-KB warm before reporting
            # ready - double-warming would just contend for the CPU.
            from rag.retriever import Retriever

            if isinstance(self.retriever, Retriever) and self.retriever.ready:
                self.retriever.warm_vectors(background=False)
        # Situations are independent of the FAISS index: a missing index must
        # not silently disable the scripted fast path as well.
        if self.situations is not None and not self.situations.rows:
            self.situations.load()
        if self.llm is None and self.memory_manager is not None:
            from llm import select_backend

            self.llm = select_backend(self.cfg)
        if self.tts is not None:
            self.tts.start()
            prewarm = getattr(self.tts, "prewarm", None)
            if callable(prewarm):
                # Synthesize the short high-traffic lines once so fillers and
                # error replies answer from cache instantly.
                prewarm([*self.cfg.filler_phrases, self.cfg.fallback_reply])
        logger.info("warmup done in %.2fs", time.perf_counter() - started)

    # ------------------------------------------------------------------
    def process_text(self, user_text: str) -> str:
        """Full think+speak cycle for one user utterance. Returns reply text."""
        assert self.memory_manager is not None
        from asr_correct import correct_transcript

        user_text = correct_transcript(user_text)
        idle_s = self.memory_manager.idle_seconds(self.session_id)
        if idle_s is not None and idle_s > self.cfg.session_idle_reset_min * 60:
            # Kiosk hygiene: a new visitor after an idle gap must not inherit
            # the previous child's facts/summary. Checked before get() so the
            # touch() inside get() cannot mask the idle gap.
            logger.info(
                "session idle %.1f min - rotating to a fresh session", idle_s / 60
            )
            self.session_id = self._new_session_id()
            # A new visitor must not inherit the previous thread's evidence.
            self._parked_docs = {"sim": 0.0}
        memory = self.memory_manager.get(self.session_id)
        memory.add_user(user_text)
        started = time.perf_counter()
        trace = self.tracer.start(self.session_id, user_text)

        # --- embed once: all downstream consumers share this vector ---
        q_vec = None
        embedder = getattr(self.retriever, "embedder", None) or getattr(
            self.situations, "embedder", None
        )
        if embedder is not None:
            try:
                q_vec = embedder.encode_query(user_text)
            except Exception:
                logger.debug("embed-once failed; consumers will embed individually")

        try:
            situation = (
                self.situations.match(user_text, q_vec=q_vec)
                if (self.situations and self.situations.ready)
                else None
            )
            if situation is not None and situation.answer:
                trace.mark("situation")
                reply = situation.answer
                path = "situation"
                if self.answer_cache:
                    self.answer_cache.store(user_text, reply, q_vec=q_vec)
                self._queue_speech(reply)
            else:
                # A guidance-only situation match steers the LLM answer.
                guidance = (
                    situation.guidance
                    if (situation is not None and situation.guidance)
                    else None
                )
                cached_reply = (
                    self.answer_cache.lookup(user_text, q_vec=q_vec)
                    if self.answer_cache
                    else None
                )
                if cached_reply is not None:
                    trace.mark("answer_cache")
                    reply, path = cached_reply, "answer-cache"
                    # Replayed answers still count as "shown" for dedup.
                    if self.answer_cache.last_hit_chunk_ids:
                        memory.mark_chunks_shown(self.answer_cache.last_hit_chunk_ids)
                    self._queue_speech(reply)
                else:
                    reply, path = self._generate_reply(
                        user_text,
                        memory,
                        trace=trace,
                        q_vec=q_vec,
                        guidance=guidance,
                    )
        except Exception:
            logger.exception("turn failed")
            reply = self.cfg.fallback_reply
            path = "fallback"
            self._queue_speech(reply)

        memory.add_bot_reply(reply)
        self._maybe_summarize(memory)
        elapsed_s = time.perf_counter() - started
        ttft_s = getattr(trace, "_extra", {}).get("ttft_s")
        doc_count = getattr(trace, "_extra", {}).get("docs", 0)
        trace.finish(path=path, reply_chars=len(reply))
        self.conv_log.log_turn(
            session_id=self.session_id,
            user_text=user_text,
            reply=reply,
            path=path,
            elapsed_s=elapsed_s,
            ttft_s=ttft_s,
            docs=doc_count,
        )
        logger.info(
            "turn done in %.2fs via %s | user=%r reply=%r",
            elapsed_s,
            path,
            user_text[:60],
            reply[:80],
        )
        return reply

    # ------------------------------------------------------------------
    def _queue_speech(self, text: str) -> None:
        """Queue a complete (non-streamed) text for speaking."""
        if self.tts is None or self.tts.disabled or not text.strip():
            return
        self.tts.reset_reply_bookkeeping()
        splitter = self._make_splitter()
        sentences = splitter.push(text) + splitter.flush()
        for sentence in sentences:
            self.tts.submit(sentence)

    def _make_splitter(self) -> SentenceSplitter:
        return SentenceSplitter(
            max_chars=self.cfg.tts_max_chunk_chars,
            early_first_clause=self.cfg.ttfa_first_clause,
        )

    # ------------------------------------------------------------------
    def _memory_ctx(self, memory, user_text: str) -> dict:
        """Serializable snapshot for service-side tool retrieval."""
        return {
            "topics": memory.last_topics(),
            "looks_like_followup": memory.looks_like_followup(user_text),
            "seen_chunk_ids": sorted(
                memory.recently_seen_chunk_ids(self.cfg.dedup_window_turns)
            ),
        }

    def _parked_sim(self) -> float:
        """Best evidence sim from the previous tool-mode turn (audit slot)."""
        return float((getattr(self, "_parked_docs", None) or {}).get("sim", 0.0))

    def _generate_reply(
        self,
        user_text: str,
        memory,
        trace=None,
        q_vec=None,
        guidance: str | None = None,
    ) -> tuple[str, str]:
        # Circuit breaker open: answer instantly instead of stalling every
        # visitor turn against a backend that just failed repeatedly.
        remaining = self._llm_cooldown_until - time.monotonic()
        if remaining > 0:
            logger.warning(
                "LLM circuit open (%.0fs left) - instant fallback", remaining
            )
            return self.cfg.fallback_reply, "llm-circuit"

        docs: list[dict] = []
        self._turn_truncated = False
        result = None
        mode = self.cfg.retrieval_mode
        agentic = mode in ("auto", "grounded")
        speculative: dict = {"done": False, "result": None}

        def speculate() -> None:
            """Pre-run retrieval on the enriched utterance, in parallel.

            If the agent's own search query turns out similar, the parked
            result is reused (0ms); otherwise it is discarded. Purely an
            accelerator - correctness never depends on it.
            """
            try:
                if self.retriever is not None and self.retriever.ready:
                    speculative["result"] = self.retriever.retrieve(
                        user_text, memory=memory, q_vec=q_vec
                    )
            except Exception:
                logger.debug("speculative retrieval failed", exc_info=True)
            finally:
                speculative["done"] = True

        if agentic:
            # Overlap search latency with leg-1 generation.
            threading.Thread(target=speculate, name="spec-fetch", daemon=True).start()
            # Guardrail: very short follow-ups right after strong-evidence
            # turns take the deterministic pipeline path instead of trusting
            # the agent's discretion.
            if (
                self.cfg.tool_guardrail
                and self._parked_sim() >= self.cfg.evidence_sim_min
                and len(user_text.split()) <= 3
            ):
                logger.info("tool guardrail: short follow-up - pipeline fallback")
                agentic = False
        if not agentic:
            if self.retriever is not None and self.retriever.ready:
                if speculative["done"] and speculative["result"] is not None:
                    result = speculative["result"]  # already fetched for us
                else:
                    result = self.retriever.retrieve(
                        user_text,
                        memory=memory,
                        exclude_ids=set(),
                        q_vec=q_vec,
                    )
                trace.mark("retrieval")
                if result.docs:
                    docs = [
                        {"path": d.path, "text": d.text, "score": d.score}
                        for d in result.docs
                    ]
                    memory.mark_chunks_shown([d.chunk_id for d in result.docs])
            if trace is not None:
                payload = {"docs": len(docs)}
                if result is not None:
                    payload["best_sim"] = round(result.best_sim, 3)
                trace.set(**payload)

        messages, meta = build_messages(
            self.system_prompt,
            memory,
            user_text,
            docs=docs,
            history_limit=self.cfg.recent_exchanges,
            guidance=guidance,
        )
        logger.debug("prompt built: %s", meta)
        if trace is not None:
            trace.set(**meta)

        self._turn_tool_events = []
        splitter = self._make_splitter()
        deadline = Deadline(self.cfg.llm_hard_deadline_s)
        spoke_fillers = False
        got_first_token = False
        parts: list[str] = []
        filler_timer: threading.Timer | None = None

        def speak_filler() -> None:
            nonlocal spoke_fillers
            if (
                not got_first_token
                and not spoke_fillers
                and self.tts
                and not self.tts.disabled
            ):
                phrase = random.choice(self.cfg.filler_phrases)
                logger.info("TTFT slow - speaking filler %r", phrase)
                spoke_fillers = True
                self.tts.submit(phrase, tag="filler")

        def tool_executor(query: str) -> str:
            """search_kb execution - local retriever or RemoteRetriever."""
            import difflib

            from prompts import format_retrieved_block

            # Speculative hit: same intent as the pre-run query -> free.
            spec = speculative.get("result")
            if spec is not None:
                q_clean = query.lower().strip()
                spec_clean = spec.query_used.lower().strip()
                ratio = difflib.SequenceMatcher(None, q_clean, spec_clean).ratio()
                q_words = set(q_clean.split())
                spec_words = set(spec_clean.split())
                overlap = (
                    len(q_words & spec_words) / max(1, min(len(q_words), len(spec_words)))
                    if q_words and spec_words
                    else 0.0
                )
                if ratio >= 0.6 or overlap >= 0.7:
                    logger.info(
                        "tool search reuse (spec ratio=%.2f, overlap=%.2f): %r",
                        ratio,
                        overlap,
                        query[:60],
                    )
                    result = spec
                else:
                    with budget("tool.search", 2.0):
                        result = (
                            self.retriever.retrieve(query, memory=memory)
                            if self.retriever is not None
                            and self.retriever.ready
                            else None
                        )
            else:
                with budget("tool.search", 2.0):
                    result = (
                        self.retriever.retrieve(query, memory=memory)
                        if self.retriever is not None and self.retriever.ready
                        else None
                    )
            if trace is not None:
                trace.mark("tool-search")
            if result is None:
                logger.warning("tool search unavailable - empty result")
                return ""
            memory.mark_chunks_shown([d.chunk_id for d in result.docs])
            self._turn_tool_events.append(
                {
                    "query": query,
                    "docs": len(result.docs),
                    "best_sim": round(result.best_sim, 3),
                }
            )
            logger.info(
                "tool search %r -> %d docs (best_sim=%.3f)",
                query[:60], len(result.docs), result.best_sim,
            )
            if not result.docs:
                return ""
            return format_retrieved_block(
                [
                    {"path": d.path, "text": d.text, "score": d.score}
                    for d in result.docs
                ]
            )[:8000]

        try:
            assert self.llm is not None
            if self.tts is not None:
                self.tts.reset_reply_bookkeeping()
            stream = self.llm.stream(
                messages,
                temperature=self.cfg.llm_temperature,
                # Tool legs carry thought signatures + a rewrite pass; the
                # pipeline's reply-sized budget starves them into one word.
                max_tokens=(
                    max(self.cfg.tool_max_tokens, self.cfg.llm_max_tokens)
                    if agentic
                    else self.cfg.llm_max_tokens
                ),
                tools=agentic,
                memory_ctx=self._memory_ctx(memory, user_text) if agentic else None,
                tool_executor=tool_executor if agentic else None,
                force_search=(mode == "grounded"),
            )
            filler_timer = threading.Timer(self.cfg.ttft_filler_after_s, speak_filler)
            filler_timer.daemon = True
            filler_timer.start()

            for chunk in stream:
                if got_first_token is False:
                    got_first_token = True
                    if filler_timer:
                        filler_timer.cancel()
                    if trace is not None:
                        trace.mark("llm_ttft")
                        trace.set(ttft_s=round(deadline.elapsed, 3))
                    logger.info("first token after %.2fs", deadline.elapsed)
                for sentence in splitter.push(chunk):
                    if self.tts:
                        self.tts.submit(sentence)
                    parts.append(sentence)
                if deadline.expired:
                    logger.warning("LLM hard deadline hit (%.2fs)", deadline.elapsed)
                    self._turn_truncated = True
                    break
                if self._barge_in.is_set():
                    logger.info("generation interrupted by barge-in")
                    self._turn_truncated = True
                    break
            tail = splitter.flush()
            for sentence in tail:
                if self.tts:
                    self.tts.submit(sentence)
                parts.append(sentence)
        except Exception:
            tripped = self.llm_failures.record_failure()
            if tripped:
                self._llm_cooldown_until = time.monotonic() + self.cfg.llm_cooldown_s
                logger.critical(
                    "LLM failed %d times in a row - circuit OPEN for %.0fs",
                    self.llm_failures.count,
                    self.cfg.llm_cooldown_s,
                )
            raise
        else:
            # Deadline/barge-in exits are normal completions: the backend
            # answered, so the failure counter resets.
            self.llm_failures.record_success()
        finally:
            if filler_timer:
                filler_timer.cancel()

        reply = " ".join(p.strip() for p in parts).strip()
        path = "llm" if docs else "llm-nodocs"

        # Tool-mode bookkeeping: rich events come from the local executor;
        # remote mode falls back to the done-event fields parsed by
        # RemoteLLM. Updates the parked-evidence slot and audits skips.
        # Answer-cache stays pipeline-only: its self-contained check needs
        # the pre-retrieval query verdict.
        if mode in ("auto", "grounded"):
            events = self._turn_tool_events or [
                {
                    "query": str(e.get("query", ""))[:80],
                    "docs": int(e.get("docs", 0) or 0),
                    "best_sim": round(float(e.get("best_sim", 0.0)), 3),
                }
                for e in getattr(self.llm, "last_tool_events", [])
            ]
            skipped = bool(getattr(self.llm, "tool_skipped", True))
            if events:
                # Single source of truth for docs/best_sim on tool turns
                # (feeds conv_log's doc_count too).
                best = max(int(e.get("docs", 0) or 0) for e in events)
                sim = max(float(e.get("best_sim", 0.0)) for e in events)
                if best > 0:
                    path = "llm"
                if trace is not None:
                    trace.mark("tool-search")
                    trace.set(docs=best, best_sim=round(sim, 3),
                              tool_events=events)
            if skipped:
                logger.info(
                    "agent skipped search (parked_sim=%.3f) - answering from thread",
                    self._parked_sim(),
                )
                if trace is not None:
                    trace.mark("tool-skip")
                    trace.set(parked_sim=round(self._parked_sim(), 3))
            if events:
                sim_now = max(
                    float(e.get("best_sim", 0.0)) for e in events
                )
                self._parked_docs = {"sim": sim_now}

        if (
            self.answer_cache
            and path == "llm"
            and not self._turn_truncated
            and result is not None
            and result.query_used == user_text
        ):
            # Self-contained + doc-grounded + fully spoken: safe to replay for
            # the next visitor who asks the same thing.
            self.answer_cache.store(
                user_text,
                reply,
                q_vec=q_vec,
                chunk_ids=[d.chunk_id for d in result.docs],
            )
        return (reply or self.cfg.fallback_reply), path

    # ------------------------------------------------------------------
    def _maybe_summarize(self, memory) -> None:
        if self.llm is None or self.memory_manager is None:
            return
        thread = self.memory_manager.maybe_summarize_async(
            memory,
            complete_fn=lambda prompt, **kw: self.llm.complete(
                [{"role": "user", "content": prompt}], **kw
            ),
        )
        if thread:
            self._summarizer_threads.append(thread)

    def join_background_work(self, timeout: float = 5.0) -> None:
        for thread in self._summarizer_threads:
            thread.join(timeout=timeout)
        self._summarizer_threads.clear()

    # ------------------------------------------------------------------
    # Operator-facing hint repeated after every finished turn so each turn
    # reads as a closed block in the console.
    TURN_HINT = (
        "Nhấn ENTER để bắt đầu nói, nhấn ENTER lần nữa để gửi.\n"
        "Trong lúc ông nói: nhấn ENTER để chen ngang. Ctrl+C để thoát."
    )
    TURN_SEP = "=" * 64

    def _print_turn_footer(self) -> None:
        print(f"\n{self.TURN_HINT}\n{self.TURN_SEP}\n")

    def run_voice(self) -> None:
        from audio import EnterKeyWatcher, MicRecorder

        watcher = EnterKeyWatcher()
        watcher.start()
        recorder = MicRecorder(self.cfg)
        print("\n=== Ông Tiến sĩ Giấy - realtime ===")
        print(f"{self.TURN_HINT}\n{self.TURN_SEP}")

        if self.stt is not None and not self.stt.ready:
            self.stt.load_async(
                callback=lambda ok: print(
                    f"[ASR] {'sẵn sàng' if ok else 'LỖI tải model - kiểm tra log'}"
                )
            )

        last_activity = time.monotonic()
        attract_idx = 0

        while True:
            try:
                if watcher.consume_press():
                    last_activity = time.monotonic()
                    if self.tts and self.tts.busy:
                        # Barge-in: cut audio now; keep only the heard prefix in
                        # history so the next turn reflects reality. The event
                        # is NOT cleared here - an in-flight LLM stream must
                        # observe it; the pre-turn clear below is the only
                        # reset point.
                        self._barge_in.set()
                        self.tts.stop()
                        heard = self.tts.heard_text(tag="reply")
                        if heard:
                            memory = self.memory_manager.get(self.session_id)
                            memory.amend_last_bot_reply(heard)
                            print(f"  (ông dừng lại, đã nghe: {heard[:60]}...)")
                        continue
                    audio = recorder.record_push_to_talk(
                        stop_check=watcher.consume_press
                    )
                    with budget("asr.transcribe", 3.0):
                        text = (
                            self.stt.transcribe(audio, SAMPLE_RATE)
                            if self.stt
                            else ""
                        )
                    if not text:
                        logger.info("empty transcript - prompting retry")
                        self._say(self.cfg.fallback_reply)
                        # Let the retry line actually play before listening
                        # again, and absorb any key-repeat backlog so a held
                        # ENTER cannot machine-gun empty turns.
                        if self.tts:
                            self.tts.wait_done(timeout=10)
                        time.sleep(0.3)
                        watcher.drain()
                        continue
                    print(f"  em nhí: {text}")
                    self._barge_in.clear()
                    reply = self.process_text(text)
                    print(f"  ông giấy: {reply}")
                    if self.tts:
                        self.tts.wait_done(timeout=self.cfg.llm_hard_deadline_s + 30)
                    logger.info("turn complete - audio finished, awaiting next ENTER")
                    self._print_turn_footer()
                    last_activity = time.monotonic()
                    continue

                # Idle attract: invite passive visitors instead of sitting mute.
                if self._attract_due(last_activity):
                    line, attract_idx = _next_attract_line(
                        self._attract_lines(), attract_idx
                    )
                    logger.info("attract invitation after idle gap")
                    self._say(line)
                    if self.tts:
                        self.tts.wait_done(timeout=60)
                    last_activity = time.monotonic()
                time.sleep(0.25)
            except KeyboardInterrupt:
                print("\nThoát.")
                break
            except Exception:
                logger.exception("voice loop iteration failed")

    # ------------------------------------------------------------------
    def _attract_due(self, last_activity: float) -> bool:
        """True when the kiosk has idled long enough to invite visitors."""
        threshold_min = float(self.cfg.attract_after_min)
        if threshold_min <= 0 or not self.cfg.attract_enabled:
            return False
        if not self._attract_lines():
            return False
        if self.tts is None or self.tts.busy or self.tts.disabled:
            return False
        if self.stt is not None and not self.stt.ready:
            return False  # cannot serve a reply to whoever walks up
        return (time.monotonic() - last_activity) >= threshold_min * 60

    def _attract_lines(self) -> list[str]:
        raw = self.cfg.attract_lines
        return [part.strip() for part in raw.split("|") if part.strip()]

    def _say(self, text: str) -> None:
        print(f"  ông giấy: {text}")
        self._queue_speech(text)
