"""Central configuration for the realtime voice RAG chatbot.

All values can be overridden via environment variables (see .env.example).
Heavy dependencies (faiss, torch, faster_whisper, edge_tts) are imported
lazily by the modules that need them — importing this file is always cheap.

Tuning philosophy:
  - Only knobs with measurable venue impact are exposed.
  - Fixed constants (model paths, sample rates) are module-level, not config.
  - The [TUNE] markers in .env.example indicate empirical knobs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dep in practice

    def load_dotenv(*args, **kwargs):  # type: ignore[misc]
        return False


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# ---- Fixed constants (not tunable, not overridable via env) ----------------
SAMPLE_RATE = 16000  # every model expects 16kHz; changing breaks ASR + smart-turn
FAISS_DIR = BASE_DIR / "data" / "faiss"
LOG_DIR = BASE_DIR / "logs"
SITUATIONS_CSV = BASE_DIR / "data" / "situations.csv"
SMART_TURN_MODEL = BASE_DIR / "models" / "smart-turn-v3.2-cpu.onnx"
GIPFORMER_DIR = BASE_DIR / "models" / "gipformer-65M-i8"


def _load_env() -> None:
    for candidate in (BASE_DIR / ".env", REPO_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_env()


def _env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    # python-dotenv keeps `KEY=   # comment` as the literal comment text;
    # no legitimate setting starts with '#', so treat that as unset.
    if value.lstrip().startswith("#"):
        return default
    return value


def _env_str(key: str, default: str) -> str:
    return _env(key, default) or default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)) or default)  # type: ignore[arg-type]
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)) or default)  # type: ignore[arg-type]
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ==================================================================
    # LLM — cloud backend (also read from GOOGLE_API_KEY)
    # ==================================================================
    llm_backend: str = _env_str("LLM_BACKEND", "auto")  # auto | gemini | mock
    gemini_model: str = _env_str("GEMINI_MODEL", "gemini-3.5-flash-lite")
    gemini_api_key: str = _env_str("GEMINI_API_KEY", _env_str("GOOGLE_API_KEY", ""))
    # Gemini 3.x: "minimal"=fastest TTFT, "low"|"medium"|"high"=slower+smarter.
    gemini_thinking_level: str | None = _env("GEMINI_THINKING_LEVEL", "minimal")
    # Optional reverse proxy / regional gateway (AI Studio is global-only).
    gemini_base_url: str = _env_str("GEMINI_BASE_URL", "")
    # [TUNE] temperature and max tokens affect answer style + latency.
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.4)
    llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 220)
    # Filler spoken if no first token by this delay; lower = less dead air.
    ttft_filler_after_s: float = _env_float("TTFT_FILLER_AFTER_S", 1.8)
    # Release the opening clause at the first comma (e.g. "Đúng, ...").
    ttfa_first_clause: bool = field(
        default_factory=lambda: _env_bool("TTFA_FIRST_CLAUSE", True)
    )
    # Hard deadline: abort generation past this; partial reply is spoken.
    llm_hard_deadline_s: float = _env_float("LLM_HARD_DEADLINE_S", 15.0)
    # Circuit breaker: after N consecutive failures, skip LLM for this long.
    llm_cooldown_s: float = _env_float("LLM_COOLDOWN_S", 120.0)

    # ==================================================================
    # ASR — gipformer-65M int8 via sherpa-onnx (default)
    # ==================================================================
    asr_backend: str = _env_str("ASR_BACKEND", "gipformer")  # gipformer | whisper
    # Legacy whisper (EraX CT2) knobs — only read when ASR_BACKEND=whisper.
    asr_model: str = _env_str("ASR_MODEL", "EraX-WoW-Turbo-V1.1")
    asr_device: str = _env_str("ASR_DEVICE", "auto")  # auto | cpu | cuda
    asr_compute_type: str = _env_str("ASR_COMPUTE_TYPE", "auto")
    asr_cpu_threads: int = _env_int("ASR_CPU_THREADS", 4)
    # Decoding knobs shared by both backends.
    asr_language: str = _env_str("ASR_LANGUAGE", "vi")
    asr_beam_size: int = _env_int("ASR_BEAM_SIZE", 5)
    asr_vad_filter: bool = field(default_factory=lambda: _env_bool("ASR_VAD_FILTER", True))
    asr_condition_on_previous_text: bool = field(
        default_factory=lambda: _env_bool("ASR_CONDITION_ON_PREVIOUS_TEXT", False)
    )
    asr_hotwords: str = _env_str(
        "ASR_HOTWORDS",
        "Trung Thu, bảo tàng, dân tộc học, tiến sĩ giấy, múa lân, rối nước,"
        " đèn ông sao, bánh dẻo, bánh nướng, tò he",
    )

    # ==================================================================
    # Capture — push-to-talk end-of-turn windows
    # ==================================================================
    # [TUNE] legacy fixed quiet window when smart-turn is off/failed.
    silence_end_ms: float = _env_float("SILENCE_END_MS", 1200.0)
    # Speech shorter than this is treated as noise and dropped.
    min_speech_ms: float = _env_float("MIN_SPEECH_MS", 250.0)
    # Hard cap on a single utterance.
    max_utterance_seconds: float = _env_float("MAX_UTTERANCE_SECONDS", 15.0)

    # ==================================================================
    # TTS — engine chain: vienneu (local) -> edge-tts (cloud) -> text-only
    # ==================================================================
    tts_enabled: bool = field(default_factory=lambda: _env_bool("TTS_ENABLED", True))
    tts_engine: str = _env_str("TTS_ENGINE", "vienneu")  # vienneu | edge
    vienneu_voice: str = _env_str("VIENEU_VOICE", "Minh Đức")
    vienneu_backend: str = _env_str("VIENEU_BACKEND", "onnx")  # onnx | auto
    # edge-tts fallback voice
    tts_voice: str = _env_str("TTS_VOICE", "vi-VN-NamMinhNeural")
    tts_rate: str = _env_str("TTS_RATE", "+10%")
    tts_cache_size: int = _env_int("TTS_CACHE_SIZE", 256)
    tts_max_consecutive_failures: int = _env_int("TTS_MAX_CONSECUTIVE_FAILURES", 3)
    # Keep playback device open across visitor gaps; reopen costs ~50-100ms.
    tts_idle_close_s: float = _env_float("TTS_IDLE_CLOSE_S", 300.0)
    # [TUNE] fillers spoken while the LLM is slow; pipe-separated in env.
    filler_phrases: list[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in _env_str(
                "FILLER_PHRASES",
                "Ờm để ông nghĩ chút nào.|Chờ ông một nhịp nhé.|Hừm, để ông nhớ lại đã.",
            ).split("|")
            if p.strip()
        ]
    )
    fallback_reply: str = _env_str(
        "FALLBACK_REPLY",
        "Ông chưa nghe rõ lắm, cháu nói lại giúp ông nhé!",
    )

    # ==================================================================
    # Retrieval — gate -> FAISS -> MMR -> char budget
    # ==================================================================
    # Embedding model: top VN-MTEB pick. Changing requires re-ingest + bench.
    embed_model: str = _env_str(
        "EMBED_MODEL", "aisingapore/SEA-LION-E5-Embedding-600M"
    )
    embed_query_prompt: str = _env_str("EMBED_QUERY_PROMPT", "Retrieval")
    # [TUNE] torch threads for query encoding; 0 = default(all cores).
    embed_threads: int = _env_int("EMBED_THREADS", 4)
    # [TUNE] retrieved-text char budget; watch prompt_chars in traces.
    context_char_budget: int = _env_int("CONTEXT_CHAR_BUDGET", 1400)
    # [TUNE] centroid similarity gate; below this, skip retrieval (small talk).
    gate_threshold: float = _env_float("GATE_THRESHOLD", 0.40)
    # [TUNE] retriever tuning — only change if bench_rag shows poor recall.
    retriever_topk_candidates: int = _env_int("RETRIEVER_TOPK_CANDIDATES", 8)
    retriever_final_docs: int = _env_int("RETRIEVER_FINAL_DOCS", 4)
    retriever_min_score: float = _env_float("RETRIEVER_MIN_SCORE", 0.30)
    retriever_mmr_lambda: float = _env_float("RETRIEVER_MMR_LAMBDA", 0.72)
    # [TUNE] per recently-shown chunk penalty to avoid repeating same docs.
    dedup_penalty: float = _env_float("DEDUP_PENALTY", 0.05)
    dedup_window_turns: int = _env_int("DEDUP_WINDOW_TURNS", 3)
    # Domain vocabulary: any of these forces retrieval regardless of gate.
    domain_keywords: list[str] = field(
        default_factory=lambda: [
            "trung thu",
            "bảo tàng",
            "dân tộc học",
            "tiến sĩ giấy",
            "tiến sỹ giấy",
            "chú cuội",
            "chị hằng",
            "thỏ ngọc",
            "đèn ông sao",
            "đèn kéo quân",
            "múa lân",
            "lân sư",
            "sư rông",
            "rối nước",
            "bánh nướng",
            "bánh dẻo",
            "mâm ngũ quả",
            "tò he",
            "phỗng đất",
            "cánh diều",
            "trống đồng",
            "ô ăn quan",
            "cà kheo",
            "hàng quán",
        ]
    )

    # ==================================================================
    # Situations — scripted fast-path (data/situations.csv)
    # ==================================================================
    situations_enabled: bool = field(
        default_factory=lambda: _env_bool("SITUATIONS_ENABLED", True)
    )
    # [TUNE] match bar; lower = more scripted answers fire.
    situations_threshold: float = _env_float("SITUATIONS_THRESHOLD", 0.86)

    # ==================================================================
    # Answer cache — semantic replay for repeated questions
    # ==================================================================
    answer_cache_enabled: bool = field(
        default_factory=lambda: _env_bool("ANSWER_CACHE_ENABLED", True)
    )
    # [TUNE] cosine bar; lower = more hits, higher wrong-context risk.
    answer_cache_similarity: float = _env_float("ANSWER_CACHE_SIMILARITY", 0.92)
    answer_cache_max_entries: int = _env_int("ANSWER_CACHE_MAX_ENTRIES", 128)
    answer_cache_ttl_min: float = _env_float("ANSWER_CACHE_TTL_MIN", 240)
    # Minimum reply length to be eligible for caching (skip short acks).
    answer_cache_min_reply_chars: int = _env_int("ANSWER_CACHE_MIN_REPLY_CHARS", 30)

    # ==================================================================
    # Memory / context — session hygiene + prompt assembly
    # ==================================================================
    recent_exchanges: int = _env_int("RECENT_EXCHANGES", 4)
    # [TUNE] how often to summarize (higher = cheaper, lower = fresher context).
    summarize_every_turns: int = _env_int("SUMMARIZE_EVERY_TURNS", 6)
    summary_max_chars: int = _env_int("SUMMARY_MAX_CHARS", 700)
    session_ttl_minutes: int = _env_int("SESSION_TTL_MINUTES", 90)
    # Idle gap after which the next visitor gets a fresh session.
    session_idle_reset_min: float = _env_float("SESSION_IDLE_RESET_MIN", 3.0)

    # ==================================================================
    # Smart turn — learned end-of-turn (models/smart-turn-v3.2-cpu.onnx)
    # ==================================================================
    smart_turn_enabled: bool = field(
        default_factory=lambda: _env_bool("SMART_TURN_ENABLED", True)
    )
    # [TUNE] quiet-pause classification threshold.
    smart_turn_threshold: float = _env_float("SMART_TURN_THRESHOLD", 0.5)
    # [TUNE] how often quiet pauses get classified (>=100ms).
    smart_turn_check_ms: float = _env_float("SMART_TURN_CHECK_MS", 400.0)
    # Hard cap on extra listening for hesitant speakers.
    smart_turn_max_extra_ms: float = _env_float("SMART_TURN_MAX_EXTRA_MS", 3500.0)

    # ==================================================================
    # Idle attract mode — invite passive visitors
    # ==================================================================
    attract_enabled: bool = field(
        default_factory=lambda: _env_bool("ATTRACT_ENABLED", True)
    )
    attract_after_min: float = _env_float("ATTRACT_AFTER_MIN", 5.0)
    attract_lines: str = _env_str(
        "ATTRACT_LINES",
        "Ê các em ơi, muốn nghe chuyện Tết Trung Thu không? Cứ hỏi ông này!|"
        "Ông biết mọi điều về đèn ông sao, múa lân đó nha, hỏi thử xem!|"
        "Đến rồi mà im re vậy hả? Hỏi ông bất cứ điều gì về Trung Thu nhé!",
    )

    # ==================================================================
    # Misc — logging / telemetry
    # ==================================================================
    log_level: str = _env_str("LOG_LEVEL", "INFO")
    verbose_console: bool = field(
        default_factory=lambda: _env_bool("VERBOSE_CONSOLE", True)
    )
    telemetry_enabled: bool = field(
        default_factory=lambda: _env_bool("TELEMETRY_ENABLED", True)
    )

    # ==================================================================
    # Microservice mode (--microservice) — component base URLs
    # ==================================================================
    asr_service_url: str = _env_str("ASR_SERVICE_URL", "http://127.0.0.1:8001")
    llm_service_url: str = _env_str("LLM_SERVICE_URL", "http://127.0.0.1:8002")
    rag_service_url: str = _env_str("RAG_SERVICE_URL", "http://127.0.0.1:8003")
    tts_service_url: str = _env_str("TTS_SERVICE_URL", "http://127.0.0.1:8004")

    def ensure_dirs(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> None:
    cfg.ensure_dirs()
    handlers: list[logging.Handler] = []
    if cfg.verbose_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        handlers.append(console)
    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)
