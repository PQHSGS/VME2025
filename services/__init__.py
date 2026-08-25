"""Microservice layer — each component runs as a separate FastAPI service.

Services:
  asr_service.py   (port 8001) — speech-to-text
  llm_service.py   (port 8002) — Gemini LLM
  rag_service.py   (port 8003) — FAISS retrieval + embedding
  tts_service.py   (port 8004) — text-to-speech

Manager orchestrates all services with health checks, timeout handling,
and hot-reload (file watcher restarts changed services).
"""
