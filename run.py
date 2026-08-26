"""Entrypoint for the realtime voice RAG chatbot.

Modes:
  python run.py                    # full voice loop (mic + TTS + LLM)
  python run.py --dev              # type text instead of talking; same brain
  python run.py --no-tts           # voice input, text-only output
  python run.py --check            # component health report, then exit
  python run.py --microservice     # same kiosk, brain split across services
                                   # (ASR/LLM/RAG/TTS on ports 8001-8004,
                                   #  hot-reload per component)
"""

from __future__ import annotations

import argparse
import logging
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime voice RAG chatbot")
    # --no-asr kept as a deprecated alias: it was behaviorally identical to
    # --dev (typed turns, no mic), so two flags for one mode invited drift.
    parser.add_argument(
        "--dev",
        "--no-asr",
        dest="dev",
        action="store_true",
        help="typed turns, no mic needed (alias: --no-asr)",
    )
    parser.add_argument("--no-tts", action="store_true", help="disable speech output")
    parser.add_argument(
        "--check", action="store_true", help="print component status and exit"
    )
    parser.add_argument(
        "--microservice",
        action="store_true",
        help=(
            "run components as local services (ports 8001-8004) with hot-reload;"
            " controller keeps mic/playback/memory and calls them over HTTP"
        ),
    )
    return parser.parse_args(argv)


def build_components(cfg, args):
    """Wire concrete implementations; heavy imports happen lazily inside."""
    from answer_cache import AnswerCache
    from audio import MicRecorder  # noqa: F401 - imported to fail fast on install issues
    from memory import MemoryManager

    memory_manager = MemoryManager(cfg)

    if args.microservice:
        # Same ConversationOrchestrator, remote component stand-ins. Heavy
        # state (embedder/index/TTS weights) lives in the services and
        # survives controller restarts; only edited components reload.
        from services.clients import (
            RemoteLLM,
            RemoteRetriever,
            RemoteSTT,
            RemoteSituations,
            build_remote_tts_player,
        )

        retriever = RemoteRetriever(cfg)
        situations = RemoteSituations(cfg)
        stt = None if args.dev else RemoteSTT(cfg)
        tts = None if args.no_tts else build_remote_tts_player(cfg)
        answer_cache = (
            AnswerCache(cfg, retriever.embedder)
            if cfg.answer_cache_enabled
            else None
        )

        from orchestrator import ConversationOrchestrator

        orch = ConversationOrchestrator(
            cfg,
            retriever=retriever,
            situations=situations,
            memory_manager=memory_manager,
            stt=stt,
            tts=tts,
            answer_cache=answer_cache,
        )
        if not args.check:
            orch.llm = RemoteLLM(cfg)
        else:
            from llm import MockBackend

            orch.llm = MockBackend()
        return orch

    # ---- monolith path --------------------------------------------------
    from asr import GipformerSTT, WhisperSTT
    from llm import select_backend
    from rag.embedder import SentenceTransformersEmbedder
    from rag.retriever import Retriever
    from rag.situations import SituationMatcher
    from tts import build_tts_player

    embedder = SentenceTransformersEmbedder(
        cfg.embed_model,
        query_prompt=cfg.embed_query_prompt,
        num_threads=cfg.embed_threads,
    )
    retriever = Retriever(cfg, embedder)
    situations = SituationMatcher(cfg, embedder)
    stt_cls = GipformerSTT if cfg.asr_backend == "gipformer" else WhisperSTT
    stt = None if args.dev else stt_cls(cfg)
    # Probe catches a broken local TTS at startup (not mid-show); --check
    # stays fast and download-free.
    tts = None if args.no_tts else build_tts_player(cfg, probe=not args.check)
    answer_cache = AnswerCache(cfg, embedder) if cfg.answer_cache_enabled else None

    from orchestrator import ConversationOrchestrator

    orch = ConversationOrchestrator(
        cfg,
        retriever=retriever,
        situations=situations,
        memory_manager=memory_manager,
        stt=stt,
        tts=tts,
        answer_cache=answer_cache,
    )
    # select_backend probes endpoints; keep it out of --check fast path
    if not args.check:
        orch.llm = select_backend(cfg)
    else:
        from llm import MockBackend

        orch.llm = MockBackend()
    return orch


def health_report(orch) -> int:
    from config import FAISS_DIR, SITUATIONS_CSV

    cfg = orch.cfg
    lines: list[tuple[str, str]] = []

    lines.append(
        (
            "config",
            f"backend={cfg.llm_backend} model={cfg.gemini_model} "
            f"gemini={'set' if cfg.gemini_api_key else 'missing'}",
        )
    )
    if cfg.asr_backend == "gipformer":
        lines.append(
            ("asr", f"gipformer-65M int8 ({cfg.asr_backend})")
        )
    else:
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            import asr

            lines.append(
                ("asr", f"device={asr.detect_device(cfg.asr_device)} model={cfg.asr_model}")
            )
        except Exception as exc:
            lines.append(("asr", f"IMPORT FAIL: {exc}"))

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = sum(1 for d in devices if d["max_input_channels"] > 0)
        outputs = sum(1 for d in devices if d["max_output_channels"] > 0)
        lines.append(("audio", f"{inputs} mic(s), {outputs} speaker(s)"))
    except Exception as exc:
        lines.append(("audio", f"FAIL: {exc}"))

    # --check never loads the index/embedder (download-free); report what is
    # actually on disk instead of "not loaded in this process".
    if orch.retriever.ready:
        lines.append(("retriever", "index loaded"))
    elif (FAISS_DIR / "index.faiss").exists():
        lines.append(
            ("retriever", f"index on disk at {FAISS_DIR} (loads at startup)")
        )
    else:
        lines.append(
            ("retriever", f"NO INDEX at {FAISS_DIR} - run `python -m rag.ingest`")
        )
    if orch.situations and orch.situations.rows:
        lines.append(("situations", f"{len(orch.situations.rows)} rows"))
    else:
        try:
            import csv as _csv

            with open(SITUATIONS_CSV, newline="", encoding="utf-8-sig") as handle:
                csv_rows = sum(
                    1
                    for row in _csv.DictReader(handle)
                    if (row.get("Câu hỏi") or "").strip()
                )
            note = "" if csv_rows else " - data/situations.csv empty or missing"
            lines.append(("situations", f"{csv_rows} rows in CSV{note}"))
        except OSError:
            lines.append(("situations", f"CSV missing at {SITUATIONS_CSV}"))
    assert orch.llm is not None
    lines.append(("llm", f"active backend = {orch.llm.name}"))
    if orch.tts is None:
        lines.append(("tts", "disabled (--no-tts)"))
    else:
        engine = orch.tts.engine_name
        lines.append(
            (
                "tts",
                ("disabled (config)" if orch.tts.disabled else "enabled")
                + f" [{engine}]",
            )
        )

    print("\n--- health ---")
    ok = True
    for name, status in lines:
        flag = "!!" if ("FAIL" in status or "NO " in status) else "ok"
        ok &= flag == "ok"
        print(f"[{flag}] {name:>10}: {status}")
    return 0 if ok else 1


def report_services(manager) -> int:
    """--check for microservice mode: probe each service's /health."""
    health = manager.health()
    print("\n--- services ---")
    ok = True
    for name, data in health.items():
        status = data.get("status", "unreachable")
        flag = "ok" if status == "ok" else "!!"
        ok &= flag == "ok"
        detail = data.get("engine") or data.get("backend") or data.get("detail") or ""
        extra = f" ({detail})" if detail else ""
        chunks = data.get("index_chunks")
        if chunks:
            extra += f" [{chunks} chunks]"
        print(f"[{flag}] {name:>6}: {status}{extra}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 stdio even when stdout is a pipe or the console codepage
    # is legacy - Vietnamese must survive every transport (PS5.1 lesson).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)
    from config import Config, setup_logging

    cfg = Config()
    setup_logging(cfg)
    cfg.ensure_dirs()

    manager = None
    if args.microservice:
        from services.manager import ServiceManager

        manager = ServiceManager(enable_reload=not args.check)
        manager.start_all()
        if args.check:
            code = report_services(manager)
            manager.stop_all()
            return code

    try:
        orch = build_components(cfg, args)
        if args.check:
            return health_report(orch)

        logging.getLogger(__name__).info("starting session %s", orch.session_id)
        orch.warmup()

        if args.dev:
            print("Dev mode: gõ câu hỏi, Enter để gửi, 'q' để thoát.")
            while True:
                try:
                    text = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in {"q", "quit", "exit"}:
                    break
                if not text:
                    continue
                reply = orch.process_text(text)
                print(f"Ã´ng giấy: {reply}\n")
            orch.join_background_work(timeout=3.0)
            if orch.tts:
                orch.tts.close()
            return 0

        orch.run_voice()
        orch.join_background_work(timeout=3.0)
        if orch.tts:
            orch.tts.close()
        return 0
    finally:
        if manager is not None:
            manager.stop_all()


if __name__ == "__main__":
    sys.exit(main())
