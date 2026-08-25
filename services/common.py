"""Shared bootstrap for the four FastAPI services.

Keeps each service module to its actual logic instead of repeating the
import-path shim, logging setup and Config() construction four times.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap(name: str):
    """Make project modules importable, set up logging, load config."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from config import Config

    return logging.getLogger(name), Config()


def init_in_background(fn, thread_name: str) -> None:
    """Run a model-loading init off the event loop so /health serves
    'loading' immediately instead of after a multi-minute cold start."""

    def worker():
        try:
            fn()
        except Exception:
            pass  # init fns capture and surface their own error state

    threading.Thread(target=worker, name=thread_name, daemon=True).start()
