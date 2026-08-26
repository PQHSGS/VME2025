"""Multi-turn retrieval chains: does context survive short follow-ups?

The seam the single-query golden set cannot see. Each CHAIN replays a
scripted conversation through the real retriever exactly as the
orchestrator drives it (memory-enriched effective queries, seen-chunk
dedup). PASS requires docs on the FINAL turn - the deepen/pivot payoff.

Chains are mode-agnostic by design: they exercise whatever RETRIEVAL_MODE
resolves to upstream of this harness (pipeline heuristics today; the
agent-written queries once tool mode is live - point --query-override at
recorded agent queries to score those separately).

Usage:
    .venv\\Scripts\\python.exe scripts/bench_chains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from memory import MemoryManager, SessionMemory
from rag.retriever import Retriever
from rag.embedder import SentenceTransformersEmbedder

# (turns..., expected_nonempty_on_last_turn)
CHAINS = [
    # Deepen: kid confirms an offer - bare "có ạ" must ride prior context.
    (
        ["đèn ông sao làm bằng gì vậy ông?", "có ạ"],
        True,
    ),
    # Elaborate request after a story.
    (
        ["kể cho ông nghe về chú Cuội đi", "sao vậy ông", "kể tiếp nhé"],
        True,
    ),
    # Pivot: brand-new topic must NOT be dragged back to the old one.
    (
        ["múa lân là gì vậy ông?", "bánh dẻo thì sao ạ?"],
        True,
    ),
    # Chit-chat between questions must not poison retrieval.
    (
        ["chào ông ạ", "rối nước là gì thế ông?"],
        True,
    ),
]


def main() -> int:
    cfg = Config()
    embedder = SentenceTransformersEmbedder(
        cfg.embed_model, query_prompt=cfg.embed_query_prompt,
        num_threads=cfg.embed_threads,
    )
    retriever = Retriever(cfg, embedder)
    if not retriever.load():
        print("no FAISS index - run rag.ingest first")
        return 1
    print("warming chunk vectors...")
    retriever.warm_vectors(background=False)

    manager = MemoryManager(cfg)
    failures = 0
    for ci, (turns, expect_docs) in enumerate(CHAINS, 1):
        session_id = f"chain{ci}"
        memory: SessionMemory = manager.get(session_id)
        line = []
        for ui, text in enumerate(turns, 1):
            memory.add_user(text)
            result = retriever.retrieve(text, memory=memory)
            memory.mark_chunks_shown([d.chunk_id for d in result.docs])
            if memory.summary == "" and memory.needs_summary(999):
                pass
            line.append(f"t{ui}:{len(result.docs)}docs/{result.best_sim:.2f}")
        last_docs = len(result.docs) > 0
        ok = (last_docs == expect_docs) if expect_docs else True
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"chain{ci} [{status}] {' | '.join(line)}")
        manager.cleanup()

    total = len(CHAINS)
    print(f"\nchains: {total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
