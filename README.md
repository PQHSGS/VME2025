# realtime-flow — Ông Tiến sĩ Giấy AI

Realtime voice RAG chatbot for the Mid-Autumn Festival kiosk at the Vietnam
Museum of Ethnology. Vietnamese-only. Cascaded streaming pipeline:

```
mic → push-to-talk capture (ENTER starts, ENTER sends) → gipformer-65M ASR (int8)
    → homophone post-filter → situation fast-path → FAISS retrieval + evidence bar
    → layered context → Gemini LLM stream → sentence splitter
    → VieNeu-TTS v3 Turbo (local) / edge-tts fallback → speaker
                                         (barge-in anywhere)
```

Design goals, in order:

1. **TTFA** (time-to-first-audio): sentence-level pipelining, TTFT fillers,
   greedy ASR, no agentic loops (one LLM round-trip per turn).
2. **Faithful context**: retrieved docs are per-turn disposable; history stays
   clean (see *Context management*).
3. **Graceful degradation**: every external dependency has a deadline and a
   fallback; TTS disables itself rather than hanging the loop.

## Layout

| Path | Role |
|---|---|
| `config.py` | all knobs, env-overridable (`.env`); `_env` defends against dotenv comment-poisoning |
| `orchestrator.py` | turn state machine: fast-path → retrieve → single streamed LLM call; embed-once for all consumers |
| `memory.py` | layered session memory (facts / rolling summary / recent window) |
| `prompts.py` + `prompts/system_prompt.md` | budgeted prompt assembly + persona |
| `rag/` | ingest, retriever (gate→FAISS→MMR→char budget), scripted situations |
| `answer_cache.py` | semantic replay cache with multi-variant rotation: repeated questions skip retrieval + LLM entirely |
| `llm.py` | backends: Gemini / mock (`select_backend` fails loud when no live backend) |
| `asr.py` | gipformer-65M int8 via sherpa-onnx (default); WhisperSTT legacy fallback |
| `asr_correct.py` | zero-latency domain-homophone filter; venue overrides via `data/asr_homophones.csv` |
| `tts.py` | engine chain: VieNeu v3 Turbo (local) → edge-tts → text-only; prefetch queue, LRU cache, barge-in stop, heard-sentence tracking |
| `tts_vienneu.py` | VieNeu wrapper (`vieneu` SDK; ONNX/CPU, threads-tunable) |
| `sentences.py` | Vietnamese-aware incremental splitter (first-clause early release) |
| `resilience.py` | deadlines, soft budgets, failure tracker (LLM circuit breaker) |
| `telemetry.py` | one JSONL span per turn → `logs/traces.jsonl`; transcripts → `conversations.jsonl` |
| `audio.py` | PTT capture (ENTER starts, ENTER sends); edge-detected ENTER watcher |
| `services/` | microservice layer: FastAPI apps on :8001-8004, Remote* clients, process manager |

## Setup

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # then set GEMINI_API_KEY

.venv\Scripts\python.exe -m rag.ingest            # ONE TIME: build data/faiss
.venv\Scripts\python.exe scripts/fetch_gipformer.py   # ONE TIME: ASR weights (~70MB)
.venv\Scripts\python.exe run.py --check           # component health report
```

Embedder + VieNeu weights download once into the HF cache on first use —
pre-download on offline boxes.

## Running

Single command (spawns all four services itself):

```bash
python run.py --microservice     # services on :8001-8004 + kiosk controller
```

Or manage services yourself and let the controller adopt them:

```bash
python -m services.asr_service --port 8001    # + llm/rag/tts on 8002-8004
python run.py --microservice                  # adopts healthy listeners
```

Controller-only flags: `--dev` (typed turns, same brain), `--no-tts`,
`--check`. Diagnostics: `scripts/smoke_services.py`, `scripts/check_audio.py`,
`scripts/bench_rag.py`, `scripts/bench_latency.py`, `scripts/bench_tts_speed.py`,
`scripts/trace_summary.py`.

Voice loop: press **ENTER**, speak, press **ENTER** again to send —
hesitating kids are never cut off by a timer.

While the bot is talking, ENTER = barge-in (cuts audio; only the
actually-heard prefix is kept in history). After `ATTRACT_AFTER_MIN` quiet
minutes the kiosk speaks a short rotating invitation.

Tests:

```bash
python -m pytest tests -q         # offline, mock-safe
ruff check .                      # lint gate (see .ruff.toml policy)
```

## Context management (why it looks like this)

Each turn assembles a fresh, budgeted user-turn payload; raw docs never enter
history, so context cannot compound across turns:

```
[system] persona + rules (byte-stable → provider prefix caching)
[user]   TÓM TẮT (rolling summary)      ← async refresh every N turns
         THÔNG TIN ĐÃ BIẾT              ← regex-extracted facts (name, likes)
         TÀI LIỆU THAM KHẢO [1..k]      ← retrieval hits, char-budgeted
         GỢI Ý TRẢ LỜI (optional)       ← operator steering from situations.csv
         HỘI THOẠI GẦN ĐÂY              ← last K verbatim exchanges
         CÂU HỎI HIỆN TẠI
[assistant] "Ông nghe rồi..."           ← role pre-ack (stripped at the
                                          Gemini protocol boundary)
```

Supporting behaviors: seen-chunk penalties prevent re-telling within a
window; short follow-ups ("nó ở đâu vậy?") are enriched with recent topics
before embedding; barge-in rewrites history to what was actually spoken.

## Latency & failure contract

Measured on the kiosk CPU: warm VieNeu synthesis runs faster than realtime
(`VIENEU_THREADS=4`, RTF ≈ 0.7–0.9), gipformer decode ≈ RTF 0.033, Gemini
TTFT ≈ 1s warm. TTS streams sentence chunks through 2 parallel synthesis
workers with an order-preserving queue; chunks are comma-split to
`TTS_MAX_CHUNK_CHARS` so synthesis stays hidden behind playback (any >0.5s
audible gap is logged for tuning). Soft budgets warn when breached:
ASR ≤3s, LLM hard deadline 15s. Failure handling:

- **LLM circuit breaker**: 3 consecutive backend failures open the circuit
  for `LLM_COOLDOWN_S` — turns answer instantly with the fallback reply
  instead of stalling; a clean reply closes it again.
- **Startup fail-loud**: `select_backend` raises when no LLM is reachable;
  the silent mock loop only exists via explicit `LLM_BACKEND=mock`.
- **Microservice resilience**: a dead service self-respawns (manager) or is
  restarted by its operator terminal; the kiosk degrades per-component
  (no docs / text-only / polite retry) instead of crashing turns.
- **TTS engine chain**: VieNeu (offline) primary, edge-tts cloud fallback,
  text-only as the last resort.
- **Session hygiene**: >`SESSION_IDLE_RESET_MIN` idle minutes rotate the
  session so the next visitor never inherits the previous child's facts.

Traces land in `logs/traces.jsonl`:

```json
{"stages_ms": {"situation": 12, "retrieval": 48,
               "llm_ttft": 610}, "path": "llm", ...}
```

## Kiosk ops notes

- Data assets live here: `data/kb/*.txt` (knowledge base),
  `data/situations.csv` (scripted Q&A + guidance column). Edit those, rerun
  `rag.ingest`; grow `data/asr_homophones.csv` from venue traces.
- TTS voice: set `VIENEU_VOICE` to one of the ~20 v3-Turbo presets
  (`python -c "from vieneu import Vieneu; print(Vieneu().list_preset_voices())"`).
- API keys come from `.env` only; `.env` is gitignored, `.env.example` is the
  documented template.
