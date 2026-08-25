"""Service manager — orchestrates microservices with hot-reload.

Spawns ASR, LLM, RAG, TTS as separate FastAPI processes on dedicated ports.
Provides the same interface as the monolithic ConversationOrchestrator but
routes each component call over HTTP. Features:

  - Health checks on every call (fail-fast per component)
  - Timeout handling per service (LLM gets longer deadline than ASR)
  - Hot-reload: file watcher detects .py changes, restarts affected service
  - Graceful degradation: ASR down -> text mode, TTS down -> text-only

Ports:
  8001 — ASR (gipformer-65M int8)
  8002 — LLM (Gemini)
  8003 — RAG (FAISS + embedder + situations)
  8004 — TTS (VieNeu / edge-tts)

Usage:
  python -m services.manager              # start all services + manager
  python -m services.manager --no-reload  # disable hot-reload
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

logger = logging.getLogger("manager")

# Service definitions: (name, module, port)
SERVICES = [
    ("asr", "services.asr_service", 8001),
    ("llm", "services.llm_service", 8002),
    ("rag", "services.rag_service", 8003),
    ("tts", "services.tts_service", 8004),
]

# Per-service timeout defaults (seconds)
TIMEOUTS = {
    "asr": 5.0,
    "llm": 16.0,  # slightly above LLM_HARD_DEADLINE_S
    "rag": 5.0,
    "tts": 10.0,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


class ServiceProcess:
    """Manages a single service subprocess."""

    def __init__(self, name: str, module: str, port: int):
        self.name = name
        self.module = module
        self.port = port
        self.process: subprocess.Popen | None = None
        self.ready = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                return  # already running
            cmd = [
                PYTHON, "-m", self.module,
                "--host", "127.0.0.1",
                "--port", str(self.port),
            ]
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.ready = False
            logger.info("[%s] started on port %d (pid=%d)", self.name, self.port, self.process.pid)

    def stop(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                logger.info("[%s] stopped", self.name)
            self.process = None
            self.ready = False

    def restart(self) -> None:
        logger.info("[%s] restarting...", self.name)
        self.stop()
        time.sleep(0.5)
        self.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def health_check(self, timeout: float = 2.0) -> bool:
        try:
            resp = httpx.get(f"{self.url}/health", timeout=timeout)
            data = resp.json()
            self.ready = data.get("status") == "ok"
            return self.ready
        except Exception:
            self.ready = False
            return False

    def wait_ready(self, timeout: float = 30.0, interval: float = 0.5) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health_check():
                return True
            time.sleep(interval)
        logger.warning("[%s] not ready after %.0fs", self.name, timeout)
        return False


class ServiceManager:
    """Orchestrates all services with health checks and hot-reload."""

    def __init__(self, enable_reload: bool = True):
        self.services: dict[str, ServiceProcess] = {}
        self._client = httpx.Client(timeout=30.0)
        self._reload_enabled = enable_reload
        self._watcher_thread: threading.Thread | None = None
        self._file_mtimes: dict[str, float] = {}
        self._running = False

        for name, module, port in SERVICES:
            self.services[name] = ServiceProcess(name, module, port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_all(self) -> None:
        """Start all services in parallel, wait for them to be ready."""
        logger.info("starting all services...")
        threads = []
        for svc in self.services.values():
            svc.start()
            t = threading.Thread(target=svc.wait_ready, daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=30.0)

        ready = sum(1 for s in self.services.values() if s.ready)
        logger.info("services ready: %d/%d", ready, len(self.services))

        if self._reload_enabled:
            self._start_watcher()

    def stop_all(self) -> None:
        self._running = False
        for svc in self.services.values():
            svc.stop()
        self._client.close()
        logger.info("all services stopped")

    # ------------------------------------------------------------------
    # Component calls (same interface as monolithic orchestrator)
    # ------------------------------------------------------------------
    def transcribe(self, audio_b64: str, sample_rate: int = 16000) -> str:
        """ASR: audio bytes -> text."""
        svc = self.services["asr"]
        if not svc.ready:
            return ""
        try:
            resp = self._client.post(
                f"{svc.url}/transcribe",
                json={"audio_b64": audio_b64, "sample_rate": sample_rate},
                timeout=TIMEOUTS["asr"],
            )
            return resp.json().get("text", "")
        except Exception as exc:
            logger.warning("ASR call failed: %s", exc)
            svc.ready = False
            return ""

    def embed(self, text: str) -> list[float]:
        """RAG: text -> query vector."""
        svc = self.services["rag"]
        if not svc.ready:
            return []
        try:
            resp = self._client.post(
                f"{svc.url}/embed",
                json={"text": text},
                timeout=TIMEOUTS["rag"],
            )
            return resp.json().get("vector", [])
        except Exception as exc:
            logger.warning("embed call failed: %s", exc)
            return []

    def retrieve(self, query: str, q_vec: list[float] | None = None, force: bool = False) -> dict:
        """RAG: query -> retrieved docs."""
        svc = self.services["rag"]
        if not svc.ready:
            return {"docs": [], "query_used": query, "elapsed_ms": 0}
        try:
            resp = self._client.post(
                f"{svc.url}/retrieve",
                json={"query": query, "q_vec": q_vec, "force": force},
                timeout=TIMEOUTS["rag"],
            )
            return resp.json()
        except Exception as exc:
            logger.warning("retrieve call failed: %s", exc)
            return {"docs": [], "query_used": query, "elapsed_ms": 0}

    def situation(self, query: str, q_vec: list[float] | None = None) -> dict:
        """RAG: query -> situation match."""
        svc = self.services["rag"]
        if not svc.ready:
            return {"matched": False}
        try:
            resp = self._client.post(
                f"{svc.url}/situation",
                json={"query": query, "q_vec": q_vec},
                timeout=TIMEOUTS["rag"],
            )
            return resp.json()
        except Exception as exc:
            logger.warning("situation call failed: %s", exc)
            return {"matched": False}

    def llm_complete(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 220) -> str:
        """LLM: messages -> full text."""
        svc = self.services["llm"]
        if not svc.ready:
            return ""
        try:
            resp = self._client.post(
                f"{svc.url}/complete",
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=TIMEOUTS["llm"],
            )
            return resp.json().get("text", "")
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            svc.ready = False
            return ""

    def llm_stream(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 220):
        """LLM: messages -> token iterator (SSE)."""
        svc = self.services["llm"]
        if not svc.ready:
            return iter([])
        try:
            resp = self._client.post(
                f"{svc.url}/stream",
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=TIMEOUTS["llm"],
                stream=True,
            )
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield line[6:]
        except Exception as exc:
            logger.warning("LLM stream failed: %s", exc)
            svc.ready = False

    def synthesize(self, text: str, tag: str = "reply") -> dict:
        """TTS: text -> audio bytes."""
        svc = self.services["tts"]
        if not svc.ready:
            return {}
        try:
            resp = self._client.post(
                f"{svc.url}/synthesize",
                json={"text": text, "tag": tag},
                timeout=TIMEOUTS["tts"],
            )
            return resp.json()
        except Exception as exc:
            logger.warning("TTS call failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Health & status
    # ------------------------------------------------------------------
    def health(self) -> dict[str, dict]:
        result = {}
        for name, svc in self.services.items():
            try:
                resp = self._client.get(f"{svc.url}/health", timeout=2.0)
                result[name] = resp.json()
            except Exception:
                result[name] = {"status": "unreachable"}
        return result

    # ------------------------------------------------------------------
    # Hot-reload: watch for .py changes, restart affected services
    # ------------------------------------------------------------------
    # Map of source files to the service that owns them
    _FILE_OWNERSHIP = {
        "asr.py": "asr",
        "llm.py": "llm",
        "tts.py": "tts",
        "tts_vienneu.py": "tts",
        "rag/retriever.py": "rag",
        "rag/situations.py": "rag",
        "rag/embedder.py": "rag",
        "smart_turn.py": "asr",
        "config.py": None,  # config change -> restart ALL services
    }

    def _start_watcher(self) -> None:
        """Background thread that watches .py files for changes."""
        self._running = True
        # Snapshot initial mtimes
        for rel_path in self._FILE_OWNERSHIP:
            full = PROJECT_ROOT / rel_path
            if full.exists():
                self._file_mtimes[rel_path] = full.stat().st_mtime

        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()
        logger.info("hot-reload watcher started")

    def _watch_loop(self) -> None:
        while self._running:
            time.sleep(2.0)
            changed_services = set()
            for rel_path, owner in self._FILE_OWNERSHIP.items():
                full = PROJECT_ROOT / rel_path
                if not full.exists():
                    continue
                mtime = full.stat().st_mtime
                if mtime != self._file_mtimes.get(rel_path):
                    self._file_mtimes[rel_path] = mtime
                    if owner is None:
                        # config.py changed -> restart all
                        changed_services.update(self.services.keys())
                    else:
                        changed_services.add(owner)
            for name in changed_services:
                svc = self.services.get(name)
                if svc:
                    logger.info("[hot-reload] %s changed -> restarting %s", rel_path, name)
                    svc.restart()
                    svc.wait_ready(timeout=30.0)


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Microservice manager")
    parser.add_argument("--no-reload", action="store_true", help="disable hot-reload watcher")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    manager = ServiceManager(enable_reload=not args.no_reload)

    def shutdown(sig, frame):
        logger.info("shutting down...")
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    manager.start_all()

    # Print status
    print("\n=== Microservice Manager ===")
    for name, svc in manager.services.items():
        status = "ok" if svc.ready else "FAIL"
        print(f"  [{status}] {name:>6}: port {svc.port}")
    print(f"\nHot-reload: {'ON' if not args.no_reload else 'OFF'}")
    print("Ctrl+C to stop all services.\n")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
