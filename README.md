# realtime-flow — Ông Tiến sĩ Giấy AI

Realtime voice RAG chatbot for the Mid-Autumn Festival kiosk at the Vietnam
Museum of Ethnology. Vietnamese-only. Cascaded streaming pipeline:

```
mic → capture + smart-turn end-of-stop → faster-whisper ASR → situation fast-path
    → gate → FAISS retrieval → layered context → Gemini LLM stream
    → sentence splitter → VieNeu-TTS (local) / edge-tts stream → speaker
                                                       (barge-in anywhere)
```

Design goals, in order:

1. **TTFA** (time-to-first-audio) < ~1s: sentence-level pipelining, TTFT
   fillers, greedy ASR, no agentic loops (one LLM round-trip per turn).
2. **Faithful context**: retrieved docs are per-turn disposable; history stays
   clean (see *Context management*).
3. **Graceful degradation**: every external dependency has a deadline and a
   fallback; TTS disables itself rather than hanging the loop.

## Layout

| Path | Role |
|---|---|
| `config.py` | all knobs, env-overridable (`​.env`) |
| `orchestrator.py` | turn state machine: fast-path → retrieve → single streamed LLM call |
| `memory.py` | layered session memory (facts / rolling summary / recent window) |
| `prompts.py` + `prompts/system_prompt.md` | strict budgeted prompt assembly |
| `rag/` | ingest, retriever (gate→FAISS→MMR→char budget), scripted situations |
| `llm.py` | backends: Gemini / mock (`select_backend` fails loud when no live backend) |
| `asr.py` | EraX-WoW-Turbo CT2, greedy + hotwords, lazy load |
| `tts.py` | engine chain: VieNeu-TTS v3 Turbo (local) → edge-tts → text-only; prefetch queue, LRU cache, barge-in stop, heard-sentence tracking |
| `tts_vienneu.py` | VieNeu-TTS v3 Turbo wrapper (`vieneu` SDK; ONNX/CPU int8 or PyTorch/GPU) |
| `sentences.py` | Vietnamese-aware incremental splitter |
| `resilience.py` | deadlines, soft budgets, failure tracker (LLM circuit breaker) |
| `telemetry.py` | one JSONL span per turn → `logs/traces.jsonl` |
| `audio.py` | push-to-talk capture with auto end-of-turn (smart-turn ONNX + silence fallback), noise-floor calibration |
| `smart_turn.py` | Smart Turn v3.2 end-of-turn classifier — finishes turns ~400ms after speech stops, waits out mid-sentence pauses |
| `answer_cache.py` | semantic replay cache: repeated questions skip retrieval + LLM entirely |

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env          # then set GEMINI_API_KEY

python -m rag.ingest            # ONE TIME: build data/faiss from data/kb/*.txt
                                # (needs the embedder download; run in deploy env)
python run.py --check           # component health report
```

First voice run downloads the ASR model (`erax-ai/EraX-WoW-Turbo-V1.1-CT2`)
to the HF cache. For an offline box, pre-download once or point `ASR_MODEL`
at a local CT2 directory.

## Running

```bash
python run.py            # voice loop (mic + speaker)
python run.py --dev      # typed turns, same brain — use this while tuning
python run.py --no-tts   # voice in, text out
python run.py --check    # status and exit
```

Voice loop: press **ENTER**, speak; a learned end-of-turn model ends the turn
~400ms after you stop (a fixed ~1.2s silence window is the fallback), or press
second ENTER. While the bot is talking, ENTER = barge-in (cuts audio; only
the actually-heard prefix is kept in history). After `ATTRACT_AFTER_MIN`
quiet minutes the kiosk speaks a short rotating invitation so passive
visitors discover it.

Tests / benchmarks:

```bash
python -m pytest tests -q
python scripts/bench_rag.py        # hit@k + MRR vs golden QA (index required)
python scripts/bench_latency.py    # per-stage latency percentiles (mock-safe)
```

## Context management (why it looks like this)

Each turn assembles a fresh, budgeted user-turn payload; raw docs never enter
history, so context cannot compound across turns:

```
[system] persona + rules (byte-stable → provider prefix caching)
[user]   TÓM TẮT (rolling summary)      ← async refresh every N turns
         THÔNG TIN ĐÃ BIẾT              ← regex-extracted facts (name, likes)
         TÀI LIỆU THAM KHẢO [1..k]      ← retrieval hits, char-budgeted
         HỘI THOẠI GẦN ĐÂY              ← last K verbatim exchanges
         CÂU HỎI HIỆN TẠI
[assistant] "Ông nghe rồi..."           ← role pre-ack, keeps model in character
```

Supporting behaviors: seen-chunk penalties prevent re-telling within a
window; short follow-ups ("nó ở đâu vậy?") are enriched with recent topics
before embedding; barge-in rewrites history to what was actually spoken.

## Latency & failure contract

Soft budgets log warnings when breached: ASR ≤3s, LLM hard deadline 15s.
Repeated questions replay instantly from the semantic answer cache. If no
token by `TTFT_FILLER_AFTER_S`, a random filler
line is spoken (pre-synthesized at warmup). Failure handling:

- **LLM circuit breaker**: 3 consecutive backend failures open the circuit
  for `LLM_COOLDOWN_S` — turns answer instantly with the fallback reply
  instead of stalling; a clean reply closes it again.
- **Startup fail-loud**: `select_backend` raises when no LLM is reachable;
  the silent mock loop only exists via explicit `LLM_BACKEND=mock`.
- **TTS engine chain**: VieNeu (offline) probed at startup, edge-tts as
  automatic cloud fallback, text-only as the last resort.
- **Session hygiene**: >`SESSION_IDLE_RESET_MIN` idle minutes rotate the
  session so the next visitor never inherits the previous child's facts.

Traces land in `logs/traces.jsonl`:

```json
{"stages_ms": {"situation": 12, "retrieval": 48,
               "llm_ttft": 610}, "path": "llm", ...}
```

## Kiosk ops notes

- Data assets live here: `data/kb/trung_thu.txt` (knowledge base),
  `data/situations.csv` (scripted Q&A). Edit those, rerun `rag.ingest`.
- TTS voice: set `VIENEU_VOICE` to one of the ~20 v3-Turbo presets
  (`python -c "from vieneu import Vieneu; print(Vieneu().list_preset_voices())"`).
  First run downloads the model to the HF cache — pre-download on offline boxes.
- The legacy folders (*Current Flow / API FLow / Enhanced Flow*) were retired;
  their prompts/questions/knowledge survive under `data/` and `prompts/`.
- API keys come from `.env` only. The keys hardcoded in the legacy files are
  burned into git-less disk history — treat them as compromised regardless.
