"""Service manager — lifecycle supervisor for the microservice layer.

Spawns ASR/LLM/RAG/TTS as separate FastAPI processes on dedicated ports and
keeps them alive. Component CALLS live in services.clients (protocol-matching
stand-ins injected into the orchestrator); this module only manages
processes:

  - spawn + wait-ready (parallel), or ADOPT an already-running healthy
    service on the same port (rerun after a crash without orphaning)
  - child stdout/stderr -> logs/services/<name>.log (never PIPE: an unread
    pipe buffer silently freezes the child once full)
  - crash respawn with capped backoff so a mid-show segfault self-heals
  - hot-reload watcher: edit a component's .py -> only that service restarts

Ports:
  8001 ASR (gipformer int8) | 8002 LLM (Gemini) | 8003 RAG | 8004 TTS

Usage:
  python -m services.manager [--no-reload] [--only asr,llm]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

logger = logging.getLogger("manager")

SERVICES = [
    ("asr", "services.asr_service", 8001),
    ("llm", "services.llm_service", 8002),
    ("rag", "services.rag_service", 8003),
    ("tts", "services.tts_service", 8004),
]

# Cold-start readiness budgets (torch/transformers import + weight load can
# take 1-4 minutes on the kiosk's CPU; never block startup shorter than this).
READY_TIMEOUT_S = {
    "asr": 120.0,
    "llm": 90.0,
    "rag": 300.0,
    "tts": 180.0,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
LOG_DIR = PROJECT_ROOT / "logs" / "services"

# Source file -> owning service; None means "restart everything".
_FILE_OWNERSHIP = {
    "asr.py": "asr",
    "services/asr_service.py": "asr",
    "llm.py": "llm",
    "services/llm_service.py": "llm",
    "tts.py": "tts",
    "tts_vienneu.py": "tts",
    "services/tts_service.py": "tts",
    "rag/retriever.py": "rag",
    "rag/situations.py": "rag",
    "rag/embedder.py": "rag",
    "services/rag_service.py": "rag",
    "config.py": None,  # config feeds every service
}


class ServiceProcess:
    """One service subprocess (or an adopted external one)."""

    def __init__(self, name: str, module: str, port: int):
        self.name = name
        self.module = module
        self.port = int(port)
        self.ready_timeout = READY_TIMEOUT_S.get(name, 60.0)
        self.process: subprocess.Popen | None = None
        self.external = False  # adopted pre-existing listener
        self.ready = False
        self._lock = threading.Lock()
        self._stopping = False
        self._log_fh = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------------
    def _health_ok(self, timeout: float = 2.0) -> bool:
        try:
            data = httpx.get(f"{self.url}/health", timeout=timeout).json()
            self.ready = data.get("status") == "ok"
            return self.ready
        except Exception:
            self.ready = False
            return False

    def _open_log(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(  # noqa: SIM115 - lifetime == process lifetime
            LOG_DIR / f"{self.name}.log", "ab", buffering=0
        )

    def start(self) -> bool:
        """Spawn the child unless a healthy service already listens (adopt)."""
        with self._lock:
            if self.process and self.process.poll() is None:
                return True
            if self._health_ok():
                logger.info("[%s] adopting existing healthy service", self.name)
                self.external = True
                return True
            self._stopping = False
            self.external = False
            if self._log_fh is None:
                self._open_log()
            self._log_fh.write(
                f"\n===== boot {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{self.module} on :{self.port} via {PYTHON} =====\n".encode()
            )
            self.process = subprocess.Popen(
                [PYTHON, "-m", self.module, "--host", "127.0.0.1",
                 "--port", str(self.port)],
                cwd=str(PROJECT_ROOT),
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            logger.info(
                "[%s] spawned pid=%d port=%d", self.name, self.process.pid, self.port
            )
            return True

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            proc, self.process = self.process, None
            self.ready = False
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            logger.info("[%s] stopped", self.name)

    def restart(self) -> None:
        logger.info("[%s] restarting...", self.name)
        if self.external:
            # Adopted processes belong to us now; recycle them too.
            self.external = False
        self.stop()
        time.sleep(0.5)
        self.start()

    # ------------------------------------------------------------------
    def wait_ready(self, timeout: float | None = None, interval: float = 0.5) -> bool:
        timeout = timeout or self.ready_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._health_ok():
                return True
            time.sleep(interval)
        logger.warning(
            "[%s] not ready after %.0fs (see logs/services/%s.log)",
            self.name, timeout, self.name,
        )
        return False

    def crashed(self) -> bool:
        return (
            not self.external
            and self.process is not None
            and self.process.poll() is not None
            and not self._stopping
        )


class ServiceManager:
    """Supervises all services: startup order, crash respawn, hot-reload."""

    def __init__(self, enable_reload: bool = True):
        self.services: dict[str, ServiceProcess] = {
            name: ServiceProcess(name, module, port) for name, module, port in SERVICES
        }
        self.enable_reload = enable_reload
        self._running = False
        self._watcher_thread: threading.Thread | None = None
        self._supervisor_thread: threading.Thread | None = None
        self._file_mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    def start_all(self, only: list[str] | None = None) -> None:
        selected = [
            self.services[n] for n in (only or list(self.services)) if n in self.services
        ]
        logger.info("starting %d services...", len(selected))
        for svc in selected:
            svc.start()
        threads = [
            threading.Thread(target=svc.wait_ready, daemon=True) for svc in selected
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ready = sum(1 for s in selected if s.ready)
        logger.info("services ready: %d/%d", ready, len(selected))
        self._start_supervisor(selected)
        if self.enable_reload:
            self._start_watcher()

    def stop_all(self) -> None:
        self._running = False
        for svc in self.services.values():
            svc.stop()
        logger.info("all services stopped")

    def health(self) -> dict[str, dict]:
        result = {}
        for name, svc in self.services.items():
            try:
                result[name] = httpx.get(f"{svc.url}/health", timeout=2.0).json()
            except Exception:
                result[name] = {"status": "unreachable"}
        return result

    # ------------------------------------------------------------------
    # Crash respawn
    # ------------------------------------------------------------------
    def _start_supervisor(self, watched: list[ServiceProcess]) -> None:
        self._running = True

        def loop():
            backoff = {s.name: 0 for s in watched}
            while self._running:
                time.sleep(3.0)
                for svc in watched:
                    if not svc.crashed():
                        continue
                    code = svc.process.returncode
                    backoff[svc.name] = min(backoff[svc.name] + 1, 5)
                    delay = 2.0 * backoff[svc.name]
                    logger.error(
                        "[%s] died (exit=%s) - respawn in %.0fs", svc.name, code, delay
                    )
                    time.sleep(delay)
                    if not self._running:
                        break
                    svc.start()
                    threading.Thread(
                        target=svc.wait_ready, daemon=True
                    ).start()

        self._supervisor_thread = threading.Thread(target=loop, daemon=True)
        self._supervisor_thread.start()

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        for rel in _FILE_OWNERSHIP:
            path = PROJECT_ROOT / rel
            if path.exists():
                self._file_mtimes[rel] = path.stat().st_mtime
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, name="reload-watch", daemon=True
        )
        self._watcher_thread.start()
        logger.info("hot-reload watcher started")

    def _watch_loop(self) -> None:
        while self._running:
            time.sleep(2.0)
            changed: set[str] = set()
            changed_all = False
            for rel, owner in list(_FILE_OWNERSHIP.items()):
                path = PROJECT_ROOT / rel
                if not path.exists():
                    continue
                mtime = path.stat().st_mtime
                if mtime == self._file_mtimes.get(rel):
                    continue
                self._file_mtimes[rel] = mtime
                logger.info("[hot-reload] %s changed", rel)
                if owner is None:
                    changed_all = True
                elif not changed_all:
                    changed.add(owner)
            # Never hijack externally-managed processes (operator runs them
            # in their own terminals): changing their code is THEIR restart.
            targets = []
            for name in list(self.services) if changed_all else sorted(changed):
                svc = self.services.get(name)
                if svc is None:
                    continue
                if svc.external:
                    logger.info(
                        "[hot-reload] %s is operator-managed - restart it "
                        "yourself (rerun its terminal)",
                        name,
                    )
                    continue
                targets.append(name)
            for name in targets:
                svc = self.services[name]
                svc.restart()
                threading.Thread(target=svc.wait_ready, daemon=True).start()


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Microservice manager")
    parser.add_argument("--no-reload", action="store_true", help="disable hot-reload")
    parser.add_argument(
        "--only", default="", help="comma list to start a subset (e.g. rag,llm)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    manager = ServiceManager(enable_reload=not args.no_reload)
    manager.start_all(
        only=[p.strip() for p in args.only.split(",") if p.strip()] or None
    )

    print("\n=== Microservice Manager ===")
    for name, svc in manager.services.items():
        status = "ok" if svc.ready else "FAIL"
        print(f"  [{status}] {name:>6}: {svc.url}  (logs/services/{name}.log)")
    print(f"\nHot-reload: {'OFF' if args.no_reload else 'ON'}")
    print("Ctrl+C stops all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
