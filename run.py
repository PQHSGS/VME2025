"""Entrypoint for the realtime voice RAG chatbot.

Modes:
  python run.py                 # full voice loop (mic + TTS + LLM)
  python run.py --dev           # type text instead of talking; same brain
  python run.py --no-tts        # voice input, text-only output
  python run.py --check         # component health report, then exit
"""

from __future__ import annotations

import argparse
import logging
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime voice RAG chatbot")
    parser.add_argument(
        "--dev", action="store_true", help="keyboard-driven mode; no mic needed"
    )
    parser.add_argument("--no-tts", action="store_true", help="disable speech output")
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="disable speech input (implies typed turns)",
    )
    parser.add_argument(
        "--check", action="store_true", help="print component status and exit"
    )
    parser.add_argument(
        "--microservice",
        action="store_true",
        help="run as microservices (ASR/LLM/RAG/TTS on separate ports with hot-reload)",
    )
    return parser.parse_args(argv)


def build_components(cfg, args):
    """Wire concrete implementations; heavy imports happen lazily inside."""
    from answer_cache import AnswerCache
    from asr import GipformerSTT, WhisperSTT
    from audio import MicRecorder  # noqa: F401 - imported to fail fast on install issues
    from config import FAISS_DIR, SITUATIONS_CSV
    from llm import select_backend
    from memory import MemoryManager
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
    memory_manager = MemoryManager(cfg)
    stt_cls = GipformerSTT if cfg.asr_backend == "gipformer" else WhisperSTT
    stt = None if args.no_asr or args.dev else stt_cls(cfg)
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
        engine = getattr(orch.tts, "engine_name", "edge-tts")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from config import Config, setup_logging

    cfg = Config()
    setup_logging(cfg)
    cfg.ensure_dirs()

    # --check and --microservice don't need the full monolithic build
    if args.check:
        orch = build_components(cfg, args)
        return health_report(orch)

    if args.microservice:
        import signal
        import time as _time

        from services.manager import ServiceManager

        manager = ServiceManager(enable_reload=True)

        def shutdown(sig, frame):
            print("\nStopping services...")
            manager.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        manager.start_all()
        print("\n=== Microservice Manager ===")
        for name, svc in manager.services.items():
            status = "ok" if svc.ready else "FAIL"
            print(f"  [{status}] {name:>6}: port {svc.port}")
        print("\nHot-reload: ON (edit .py files, service auto-restarts)")
        print("Ctrl+C to stop all services.\n")
        try:
            while True:
                _time.sleep(1)
        except KeyboardInterrupt:
            shutdown(None, None)

    orch = build_components(cfg, args)
    logging.getLogger(__name__).info("starting session %s", orch.session_id)
    orch.warmup()

    if args.dev or args.no_asr:
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
            print(f"ông giấy: {reply}\n")
        orch.join_background_work(timeout=3.0)
        if orch.tts:
            orch.tts.close()
        return 0

    orch.run_voice()
    orch.join_background_work(timeout=3.0)
    if orch.tts:
        orch.tts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
