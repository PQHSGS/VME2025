"""Config regression: fields consumed by audio.py / asr.py / run.py must
exist on the production Config (tests elsewhere use ad-hoc namespaces and
would not catch their removal)."""

import config


def test_capture_fields_exist():
    cfg = config.Config()
    assert cfg.min_speech_ms > 0
    assert cfg.max_utterance_seconds > 0


def test_asr_fields_exist():
    cfg = config.Config()
    assert cfg.asr_backend in ("gipformer", "whisper")
    assert cfg.asr_cpu_threads >= 1
    assert cfg.asr_language == "vi"
    # Whisper legacy knobs referenced by asr.WhisperSTT + run.py health report
    assert isinstance(cfg.asr_model, str) and cfg.asr_model
    assert isinstance(cfg.asr_device, str)
    assert isinstance(cfg.asr_compute_type, str)


def test_service_urls_default_to_localhost():
    cfg = config.Config()
    for url in (
        cfg.asr_service_url,
        cfg.llm_service_url,
        cfg.rag_service_url,
        cfg.tts_service_url,
    ):
        assert url.startswith("http://127.0.0.1:")


def test_comment_only_env_values_are_unset(monkeypatch):
    """python-dotenv keeps `KEY=   # note` as literal comment text (v1.1);
    _env must treat such values as unset or URLs/paths get poisoned."""
    monkeypatch.setenv("GEMINI_BASE_URL", "# SET ONCE: regional proxy")
    cfg = config.Config()
    assert cfg.gemini_base_url == ""
