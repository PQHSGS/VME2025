"""Build the retrieval index from the knowledge base.

Pipeline:
  1. Parse hierarchical headers ("A." > "1." > "1.1.") into a tree.
  2. Split each section's content into sentence-aligned chunks
     (~450 chars, max ~650 - small enough for focused retrieval).
  3. Embed ``[breadcrumb] chunk_text`` so the vector carries section context.
  4. Store one normalized inner-product FAISS index + metadata JSON +
     per-top-level-section centroid vectors (used by the domain gate).

Run:  python -m rag.ingest            (uses data/kb/*.txt -> data/faiss/)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("rag.ingest")

HEADER_RE = re.compile(r"^\s*((?:[A-ZÀ-Ở-Ỹ]|(?:\d+(?:\.\d+)*)))\.\s+(.+?)\s*$")
ROMAN_RE = re.compile(r"^\s*(i{1,3}|iv|v|vi{0,3}|ix|x)\.\s+", re.IGNORECASE)

TARGET_CHARS = 450
MAX_CHARS = 650


# ----------------------------------------------------------------------
def parse_hierarchy(text: str) -> list[dict]:
    """Return nodes: {level, token, title, path(list[str]), content}."""

    def level_of(token: str) -> int:
        if token.isalpha():
            return 1
        return token.count(".") + 2

    nodes: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current: dict | None = None

    def close_current() -> None:
        nonlocal current
        if current is not None:
            nodes.append(current)
            current = None

    for raw_line in text.splitlines():
        line = raw_line.replace("\ufeff", "").strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        match = HEADER_RE.match(line)
        roman = ROMAN_RE.match(line)
        # Roman numerals stay content bullets (they are list items in the KB).
        if match and not roman:
            token, title = match.group(1).strip(), match.group(2).strip()
            level = level_of(token)
            while stack and stack[-1][0] >= level:
                stack.pop()
                close_current()
            close_current()
            stack.append((level, title))
            current = {
                "level": level,
                "token": token,
                "title": title,
                "path": [t for _, t in stack],
                "content": "",
            }
        elif current is not None:
            current["content"] = (current["content"] + " " + line).strip()
        # lines before the first header are ignored (preamble)
    close_current()
    return nodes


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_content(
    content: str, target_chars: int = TARGET_CHARS, max_chars: int = MAX_CHARS
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(content):
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= target_chars or not current:
            current = candidate
            # hard-split single monster sentences
            while len(current) > max_chars:
                chunks.append(current[:max_chars].rsplit(" ", 1)[0])
                current = current[len(chunks[-1]) :].strip()
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return [c for c in (chunk.strip() for chunk in chunks) if c]


# ----------------------------------------------------------------------
def build_documents(kb_texts: list[str]) -> list[dict]:
    docs: list[dict] = []
    doc_id = 0
    for source_name, full_text in kb_texts:
        for node in parse_hierarchy(full_text):
            breadcrumb = " > ".join(node["path"])
            for piece in chunk_content(node["content"]):
                docs.append(
                    {
                        "chunk_id": str(doc_id),
                        "source": source_name,
                        "path": breadcrumb,
                        "title": node["title"],
                        "text": piece,
                        "embed_text": f"[{breadcrumb}] {piece}",
                    }
                )
                doc_id += 1
    return docs


def build_index(docs: list[dict], embedder, out_dir: Path) -> None:
    import faiss  # lazy heavy import

    vectors = embedder.encode([d["embed_text"] for d in docs], normalize=True)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    meta = [
        {k: d[k] for k in ("chunk_id", "source", "path", "title", "text")} for d in docs
    ]
    (out_dir / "meta.json").write_text(
        json.dumps({"model_dim": dim, "docs": meta}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Per top-level-section centroids for the cheap domain gate.
    top_paths: list[str] = []
    for doc in docs:
        root = doc["path"].split(" > ")[0]
        if root not in top_paths:
            top_paths.append(root)
    centroids = np.zeros((len(top_paths), dim), dtype=np.float32)
    counts = np.zeros(len(top_paths), dtype=np.float32)
    for vec, doc in zip(vectors, docs, strict=False):
        row = top_paths.index(doc["path"].split(" > ")[0])
        centroids[row] += vec
        counts[row] += 1
    counts[counts == 0] = 1.0
    centroids /= counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids /= norms
    np.save(out_dir / "centroids.npy", centroids.astype(np.float32))
    (out_dir / "sections.json").write_text(
        json.dumps(top_paths, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "index built: %d chunks, dim=%d, sections=%s", len(docs), dim, top_paths
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    base = Path(__file__).resolve().parent.parent
    kb_dir = base / "data" / "kb"
    out_dir = base / "data" / "faiss"
    kb_files = sorted(kb_dir.glob("*.txt"))
    if not kb_files:
        logger.error("no .txt knowledge files found in %s", kb_dir)
        return 1
    kb_texts = [(f.stem, f.read_text(encoding="utf-8")) for f in kb_files]
    docs = build_documents(kb_texts)
    logger.info("parsed %d chunks from %d files", len(docs), len(kb_files))
    if not docs:
        logger.error("parser produced no chunks - check KB formatting")
        return 1

    from rag.embedder import SentenceTransformersEmbedder
    from config import Config

    cfg = Config()
    embedder = SentenceTransformersEmbedder(cfg.embed_model)
    build_index(docs, embedder, out_dir)
    print(f"OK -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
