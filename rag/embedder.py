"""Embedding backend with lazy model loading.

``SentenceTransformersEmbedder`` wraps any sentence-transformers compatible
model (default: SEA-LION-E5-Embedding-600M, see config ``embed_model``).
E5-family models are instruct-tuned, so query-side encoding goes through the
model's shipped prompt (``embed_query_prompt``) while passages stay raw.
The model is only downloaded/loaded on first ``encode`` call so importing
this module (and running the offline test-suite) never touches the network.
"""

from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np

logger = logging.getLogger("rag.embedder")


class BaseEmbedder:
    dim: int = 0

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        """Encode an asymmetric-retrieval query (may add a task prefix)."""
        return self.encode_one(text)


class SentenceTransformersEmbedder(BaseEmbedder):
    """Wraps any sentence-transformers compatible embedding model.

    E5-family models are instruction/prefix tuned: queries want the model's
    shipped "Retrieval" prompt while passages stay raw. ``embed_query_prompt``
    selects that prompt by name ("" disables); it only ever applies to
    ``encode_query`` so ingest-side vectors remain unprefixed.
    """

    def __init__(self, model_name: str, query_prompt: str = "", num_threads: int = 0):
        self.model_name = model_name
        self.query_prompt = query_prompt
        self.num_threads = num_threads
        self._model = None
        self._lock = threading.Lock()
        self.dim = 768  # placeholder; corrected after load

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import (
                        SentenceTransformer,
                    )  # heavy import

                    logger.info("loading embedding model %s ...", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
                    if self.num_threads > 0:
                        # Oversubscription is real: torch's default (= every
                        # logical core) is SLOWER than the physical-core count
                        # for single-sentence CPU encoding (measured 8t=310ms
                        # vs 4t=220ms on the kiosk box).
                        import torch

                        torch.set_num_threads(self.num_threads)
                        logger.info("embedder torch threads=%d", self.num_threads)
                    self.dim = int(self._model.get_sentence_embedding_dimension())
                    logger.info("embedding model ready (dim=%d)", self.dim)
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if not self.query_prompt:
            return self.encode_one(text)
        model = self._ensure_model()
        vec = model.encode(
            [text],
            prompt_name=self.query_prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vec[0], dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class FakeEmbedder(BaseEmbedder):
    """Deterministic hash-based vectors for offline tests/dev.

    Same string -> same unit vector; similar token overlap -> higher cosine.
    Good enough to exercise retrieval logic without any model download.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vec[index] += sign * (1.0 + (digest[5] % 100) / 100.0)
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        return (
            np.stack([self._vector(t) for t in texts])
            if texts
            else np.zeros((0, self.dim), dtype=np.float32)
        )

    def encode_one(self, text: str) -> np.ndarray:
        return self._vector(text)
