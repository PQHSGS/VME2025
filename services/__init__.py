"""Microservice layer — each component runs as a separate FastAPI service.

Layout:
  asr_service.py   (port 8001) — gipformer speech-to-text
  llm_service.py   (port 8002) — Gemini streaming/completion proxy
  rag_service.py   (port 8003) — embedder + FAISS retrieval + situations
  tts_service.py   (port 8004) — pure text->PCM synthesis (never plays)
  clients.py       — protocol-matching remote stand-ins injected into the
                     ConversationOrchestrator by run.py --microservice
  manager.py       — process supervisor: spawn/adopt, crash-respawn,
                     hot-reload watcher

The controller keeps mic capture, playback, session memory and telemetry;
heavy model state lives in the services and survives controller restarts.
"""
