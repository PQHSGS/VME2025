# AGENTS.md — realtime-flow

Guidance for AI coding agents working in this repo.

## What this is

Vietnamese-only realtime voice RAG chatbot ("Ông Tiến sĩ Giấy AI") for a
Mid-Autumn Festival kiosk at the Vietnam Museum of Ethnology. Cascaded
streaming pipeline:

```
mic → capture + smart-turn end-of-stop → gipformer ASR (int8, 30x RTF)
    → gate → FAISS retrieval → layered context → Gemini LLM stream
    → sentence splitter → VieNeu-TTS (local) / edge-tts stream → speaker
                                                       (barge-in anywhere)
```

Design goals, in order: **TTFA < ~1s**, faithful per-turn context,
graceful degradation. No agentic loops — one LLM round-trip per turn.
Do not introduce LangChain/LangGraph or agent frameworks into the hot path;
this orchestrator is deliberately a hand-rolled state machine.

## Environment — MANDATORY

All Python commands MUST run inside the project venv:

```
.venv (project root)\Scripts\python.exe        # direct (safest for agents)
.venv (project root)\Scripts\Activate.ps1      # or activate first
```

A bare `python` resolves to the system install, which is missing
faiss/torch/vieneu/sherpa-onnx and fails with confusing ImportErrors.

## Commands

```bash
.venv (project root)\Scripts\python.exe -m pip install -r requirements.txt   # copy .env.example .env first
.venv (project root)\Scripts\python.exe -m rag.ingest    # ONE TIME: build data/faiss from data/kb/*.txt
.venv (project root)\Scripts\python.exe -m pytest tests -q               # full test suite (mock-safe, offline)
.venv (project root)\Scripts\python.exe run.py --check   # component health report, no downloads
.venv (project root)\Scripts\python.exe run.py --dev     # typed turns, same brain as voice mode
.venv (project root)\Scripts\python.exe run.py           # voice loop
.venv (project root)\Scripts\python.exe run.py --microservice  # services on :8001-8004 + kiosk controller
.venv (project root)\Scripts\python.exe scripts/smoke_services.py --only rag,tts,llm  # e2e service check
.venv (project root)\Scripts\python.exe scripts/bench_rag.py         # hit@k + MRR vs golden QA (index required)
.venv (project root)\Scripts\python.exe scripts/bench_latency.py     # per-stage latency percentiles
.venv (project root)\Scripts\python.exe scripts/trace_summary.py     # summarize logs/traces.jsonl after a show-hour
```

Heavy deps (faiss, torch, faster_whisper, edge_tts, vieneu) are imported
lazily; `import config` must always stay cheap. Tests must pass without any
model download — use MockBackend / FakeEmbedder / fake TTS fakes from the
existing tests.

## Architecture map

| Path | Role |
|---|---|
| `config.py` | all knobs, env-overridable via `.env`; fixed constants (paths, sample rate) at module level |
| `orchestrator.py` | turn state machine; circuit breaker; idle session rotation; embed-once for all downstream consumers |
| `services/asr_service.py` | ASR on :8001 — gipformer transcribe over HTTP |
| `services/llm_service.py` | LLM on :8002 — Gemini stream (SSE, JSON-framed) / complete |
| `services/rag_service.py` | RAG on :8003 — embedder + FAISS retrieve (+memory ctx shim) + situations |
| `services/tts_service.py` | TTS on :8004 — pure text->PCM synth; never plays audio |
| `services/clients.py` | Remote* stand-ins matching local protocols; injected by `--microservice` |
| `services/manager.py` | process supervisor: spawn/adopt/crash-respawn/hot-reload watcher |
| `services/common.py` | shared service bootstrap (path shim, logging, cfg, bg init) |

## Conventions

- Python 3.11+ style, type hints on public functions, module docstrings
  explaining *why*, Vietnamese user-facing strings, English code/comments.
- Naming: "ASR" = the speech-to-text component/service; "stt" = the injected
  transcriber role in the orchestrator. Keep Remote* clients attribute-for-
  attribute compatible with their local twins.
- Never edit UTF-8 files via PowerShell Get-Content/Set-Content — PS5.1
  defaults to ANSI and silently mojibakes Vietnamese strings. Use the Edit
  tool or Python io.
- New config goes through `_env_*` helpers in `config.py` +
  `.env.example` documentation. In .env files keep empty values truly bare
  (`KEY=`) — python-dotenv v1.1 turns `KEY=   # comment` into the comment
  text being the value (config._env defends against leading '#').
| `memory.py` | layered session memory (facts / rolling summary / recent window) |
| `prompts.py` + `prompts/system_prompt.md` | budgeted prompt assembly |
| `rag/retriever.py` | gate → FAISS → seen-chunk penalty → MMR (chunk vectors cached) → char budget |
| `rag/situations.py` | scripted Q&A fast path from `data/situations.csv` |
| `answer_cache.py` | semantic replay cache for repeated questions (exact + cosine tiers; follow-up guarded) |
| `llm.py` | backends gemini/mock; `select_backend` fails loud |
| `asr.py` | gipformer-65M int8 via sherpa-onnx (default); WhisperSTT legacy fallback |
| `asr_correct.py` | zero-latency domain-homophone post-filter; venue overrides via `data/asr_homophones.csv` |
| `tts.py` | engine chain VieNeu → edge-tts → text-only; queue/cache/barge-in bookkeeping |
| `tts_vienneu.py` | VieNeu-TTS v3 Turbo wrapper (`vieneu` SDK) |
| `audio.py` | push-to-talk capture, smart-turn/silence auto-stop, cached noise floor |
| `smart_turn.py` | learned end-of-turn ONNX classifier (models/smart-turn-v3.2-cpu.onnx); failure → fixed window |
| `resilience.py` | Deadline, budget(), FailureTracker (LLM breaker) |
| `telemetry.py` | per-turn JSONL spans (`traces.jsonl`) + conversation transcript (`conversations.jsonl`) |

## Invariants — do not break these

1. **Retrieved docs never enter history.** They are assembled fresh each
   turn by `build_messages`; memory stores only summary/facts/recent.
2. **System prompt is byte-stable** across releases where possible
   (provider prefix caching).
3. **Everything is injectable** (`ConversationOrchestrator(cfg, llm=…,
   tts=…)`); tests/dev must exercise the exact production code path.
4. **Fail loud at startup, degrade gracefully at runtime**: missing LLM
   raises; runtime failures trip the FailureTracker circuit breaker
   (`llm-circuit` path) instead of stalling visitors 15s.
5. **Barge-in fidelity**: history records only what was actually heard
   (`heard_text(tag="reply")` + `amend_last_bot_reply`). The `_barge_in`
   event's only reset point is the pre-turn clear in `run_voice`.
6. **Session hygiene**: >`SESSION_IDLE_RESET_MIN` idle minutes rotate
   `session_id`; a new visitor must not inherit previous facts/summary.
7. **Summary replaces, never appends** — the summarizer prompt already
   folds the old summary in; `apply_summary` truncates to
   `summary_max_chars`.
8. Keep `import config` cheap; keep the offline test suite green
   (`python -m pytest tests -q`).

## Conventions

- Python 3.11+ style, type hints on public functions, module docstrings
  explaining *why*, Vietnamese user-facing strings, English code/comments.
- New config goes through `_env_*` helpers in `config.py` +
  `.env.example` documentation.
- Telemetry: new stages should `trace.mark("name")`; paths used so far:
  `situation`, `answer-cache`, `llm`, `llm-nodocs`, `llm-circuit`, `fallback`.

## Deployment notes

- TTS voice presets: `VIENEU_VOICE` (list via
  `python -c "from vieneu import Vieneu; print(Vieneu().list_preset_voices())"`).
  First runs download models (gipformer ~100MB int8, VieNeu weights, embedder)
  to the HF cache — pre-download on offline kiosks. The smart-turn ONNX model
  is bundled in `models/` (no download). Legacy EraX Whisper model is
  available via `ASR_BACKEND=whisper`.
- Targets both CPU-only boxes (ONNX/int8 paths) and small NVIDIA GPUs (4–7GB).
- `onnxruntime>=1.20` is required (smart-turn model uses IR version 10).
- gipformer int8 ONNX files: `python scripts/fetch_gipformer.py`
  (HF: g-group-ai-lab/gipformer-65M-rnnt, ~70MB total).
- Gemini transport: HTTP/2 enabled via `client_args`; `google-genai` pinned
  ≥1.46 (concurrency latency fix). httpx keepalive expires after ~5s idle, so
  the first turn after a long gap pays TCP+TLS again (~200ms) — accepted.
- ASR decodes with fixed `temperature=0`: trades away the hotter-retry escape
  hatch for rare repetition loops (those transcripts fail the gate anyway) in
  exchange for never paying the up-to-6x fallback re-decodes.

## Tuning guide (empirical, venue-first)

Knobs marked `[TUNE]` in `.env.example`. Method: change one, run a show-hour,
read `logs/traces.jsonl`.

| Knob | Watch | Adjust when |
|---|---|---|
| `SMART_TURN_THRESHOLD/CHECK_MS` | end-of-turn log lines (`p=…`) | turns cut mid-question → raise threshold; dead air → lower check_ms |
| `EVIDENCE_SIM_MIN` | `best_sim` in traces vs turns with `docs>0` | junk docs leaking into chat answers → raise; real questions answered without docs → lower (prefer low) |
| `ANSWER_CACHE_SIMILARITY` | cache-hit rate + wrong-answer complaints | wrong replayed answers → raise toward 0.95; few hits → try 0.90 |
| `CONTEXT_CHAR_BUDGET` / `RECENT_EXCHANGES` | `prompt_chars` field in traces vs `llm_ttft` | TTFT creeping up → trim budget; answers missing context → grow |
| `SITUATIONS_THRESHOLD` | situation hits logged at match time | misses on obvious scripted Qs → lower |
| `TTS_IDLE_CLOSE_S` | first-word delay after long gaps | audio device contention with other apps → shorten |

## Reference material

Cloned for study under `D:\Code\VME\references\`: VieNeu-TTS (TTS we ship),
silero-vad (VAD upgrade candidate for audio.py), gipformer (noisy-scenario
VN ASR watchlist), vietnamese-embedding-benchmark (embedding/reranker
leaderboard), pipecat + livekit-agents (framework patterns to borrow, not
adopt; the smart-turn-v3.2 ONNX we bundle ships inside pipecat), mem0
(memory-layer ideas). Do not import from them; treat as reading material
only.

**RAG watchlist** (revisit only after the KB grows beyond one file, or a
new release clearly leads VN-MTEB): embeddings — embeddinggemma-300m (47.2,
half our size), DeepX Embedding v1 (Zalo-legal SOTA, linear attention);
rerankers — Qwen3-Reranker-0.6B (beats bge-v2-m3) once candidate pools grow;
hybrid BM25 for proper-noun misses. Current pick SEA-LION-E5-600M is #3 on
the Aug-2026 VN-MTEB retrieval board (48.4 NDCG@10) and stays.
Vector store: faiss IndexFlatIP over 176x1024 vectors is exact-search
overkill already; switch to **sqlite-vec** (single-file vectors+metadata,
exact KNN to ~100k chunks) when multi-file KB makes the split meta.json
painful, and to IVF/HNSW only past ~10k chunks. No server DB at any
foreseeable scale here.
Architecture verdict (measured Aug-2026): gate→FAISS→MMR→budget hits
MRR 0.92 / p95 245ms on the golden set (after `warm_vectors()` fills the
chunk-vector cache at startup; per-turn cost ≈ ONE query encode ~200-350ms).
**GraphRAG rejected**: built for large multi-doc corpora needing entity
traversal/global summaries; our 176-chunk single-domain KB has no multi-hop
queries kids ask, offline graph indexing burns LLM calls, runtime traversal
breaks the one-round-trip invariant. **Metadata-filtered retrieval rejected**
at this N (filter≈scan); breadcrumbs already ride inside the vectors and
section centroids power the gate. The quality levers that remain are CONTENT
(KB breadth, situations.csv growth from venue logs) and measurement
(bench_rag golden set), not structure. Query-encode CPU cost is tuned via
`EMBED_THREADS` (4 beats torch's default on many-core boxes); further drops
would need ONNX-int8 export or a smaller embedder - both deferred until
venue traces show retrieval as the dominant TTFA term.

**ASR watchlist** (gipformer-65M int8 is now the default, swapped Aug-2026:
7.87% WER FLEURS-vi, RTF 0.033 on CPU — 40× faster + 6.77pts better than
EraX). Legacy EraX-WoW-Turbo V1.1 available via `ASR_BACKEND=whisper`.
Nemotron-3.5-asr-streaming-0.6b tested but NumPy/Numba incompatibility
prevented bench; streaming capability would unlock speculative speech
processing but requires GPU. Qwen3-ASR-0.6B/1.7B on watchlist. TTS: stay
on VieNeu v3 Turbo (Magpie TTS has great Vi CER but requires NIM/GPU).

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
