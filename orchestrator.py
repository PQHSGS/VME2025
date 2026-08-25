"""Conversation orchestrator - the realtime state machine.

Turn flow:
  IDLE -> RECORDING (PTT_MODE: smart auto-stop | manual toggle | hold)
        -> TRANSCRIBING (gipformer via ASR service, ~3s soft budget;
                         homophones canonicalized before the brain sees them)
        -> THINKING     (situation fast-path OR gate->retrieve->single LLM call;
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
        self.system_prompt = load_system_prompt()
        self.session_id = self._new_session_id()
        self.llm_failures = FailureTracker("llm", threshold=3)
        self._llm_cooldown_until = 0.0  # circuit breaker: monotonic deadline
        self._turn_truncated = False  # barge-in/deadline cut the reply short
        self._summarizer_threads: list[threading.Thread] = []
        self._vectors_warmed = threading.Event()
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
            if self.retriever.ready and not self._vectors_warmed.is_set():
                self._vectors_warmed.set()
                self.retriever.warm_vectors()
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
            early_first_clause=self.cfg.ttfa_first_clause,
        )

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
        if self.retriever is not None and self.retriever.ready:
            result = self.retriever.retrieve(
                user_text,
                memory=memory,
                exclude_ids=set(),  # per-turn dedup handled via seen_chunks
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

        try:
            assert self.llm is not None
            if self.tts is not None:
                self.tts.reset_reply_bookkeeping()
            stream = self.llm.stream(
                messages,
                temperature=self.cfg.llm_temperature,
                max_tokens=self.cfg.llm_max_tokens,
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
    def run_voice(self) -> None:
        from audio import EnterKeyWatcher, MicRecorder

        watcher = EnterKeyWatcher()
        watcher.start()
        recorder = MicRecorder(self.cfg)
        mode = getattr(self.cfg, "ptt_mode", "smart")
        hints = {
            "smart": "Nhấn ENTER để bắt đầu nói. Im lặng ~1.2s hoặc nhấn ENTER để kết thúc.",
            "manual": "Push-to-talk thuần: ENTER bắt đầu, chỉ ENTER kết thúc.",
            "hold": "Giữ ENTER để nói, thả tay để gửi.",
        }
        print("\n=== Ông Tiến sĩ Giấy - realtime ===")
        print(hints.get(mode, hints["smart"]))
        print("Trong lúc ông nói: nhấn ENTER để chen ngang. Ctrl+C để thoát.\n")

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
                    if mode == "hold":
                        audio = recorder.record_hold(watcher.is_down)
                    else:
                        audio = recorder.record_until_turn_end(
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
                    print(f"  ông giấy: {reply}\n")
                    if self.tts:
                        self.tts.wait_done(timeout=self.cfg.llm_hard_deadline_s + 30)
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
