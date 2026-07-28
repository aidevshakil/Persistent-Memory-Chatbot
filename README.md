# 🧠 Persistent Memory Chatbot

A Streamlit chatbot that **remembers you across sessions**. Tell it you're 25 and
love football, close the app, come back tomorrow, ask *"what sport should I try?"* —
it suggests football. Tell it you're 26 now and the old value is **replaced**, not
piled up beside the new one.

Built for the Spiral Lab *Persistent Memory Chatbot* brief.

```
you  ▸ I'm 25, I live in Dhaka and I love football
bot  ▸ Nice — Dhaka has a real football culture. …
       💾 memory: + age: 25   + city: Dhaka   + favourite_sport: football

--- app closed, process exited, restarted ---

you  ▸ What sport should I try this weekend?
       🧠 3 memories used   favourite sport: football · 0.53
bot  ▸ You already love football, so a five-a-side league would suit you…

you  ▸ Actually I'm 26 now
       💾 memory: ~ age: 26 (was: 25)
```

---

## Contents

- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [Verifying persistence](#verifying-persistence-the-2-minute-demo)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Memory strategy](#memory-strategy-summary)
- [Memory backends](#memory-backends)
- [Testing](#testing)
- [CLI](#cli)
- [Troubleshooting](#troubleshooting)

---

## What it does

| Brief requirement | How it's met |
|---|---|
| Streamlit chat UI with input + history | [`ui/streamlit_app.py`](src/memory_chatbot/ui/streamlit_app.py) — streaming replies, per-turn "memories used" panel, live sidebar of stored facts |
| Groq / Gemini / OpenAI backend | [`llm/providers.py`](src/memory_chatbot/llm/providers.py) — swappable at runtime from the sidebar, plus a key-free offline mode |
| Memory layer with disk persistence | Two backends behind one interface: built-in SQLite, or **Mem0** OSS |
| Store important details, recall them | Closed 11-category taxonomy + hybrid ranked retrieval |
| Facts survive an app restart | SQLite/WAL on disk under `data/`; verified by tests that discard all in-process state |
| New info overrides old (25 → 26) | `UNIQUE(user_id, category, key)` + a 6-step reconciliation ladder |
| Explore memory tools on the market | [`docs/memory_tools.md`](docs/memory_tools.md) — Mem0, OpenMemory, Zep/Graphiti, LangMem, Letta, Cognee and more |
| Define a memory strategy | [`docs/memory_strategy.md`](docs/memory_strategy.md) — what to store, how to recall, how conflicts resolve |

Beyond the brief: multiple isolated memory profiles, manual fact entry, per-fact
deletion, JSON export, an audit trail of superseded values, transcript
persistence, a CLI inspector, and **183 tests**.

---

## Quickstart

Requires **Python 3.11+** (developed and tested on 3.14).

```bash
# 1. install
pip install -r requirements.txt

# 2. configure — copy the template and add one API key
cp .env.example .env            # PowerShell: Copy-Item .env.example .env

# 3. run
streamlit run app.py
```

Then open http://localhost:8501.

**Which key?** Any one of these is enough:

| Provider | Get a key | Notes |
|---|---|---|
| **Groq** (default) | [console.groq.com/keys](https://console.groq.com/keys) | Free tier, fastest responses — recommended |
| Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Free tier |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Paid; also provides embeddings |

**No key at all?** The app still runs. It falls back to an offline `echo`
provider that doesn't generate prose but *does* report exactly what it recalled —
so the entire memory layer, including cross-restart persistence and overrides, can
be demonstrated with no credentials. Extraction falls back to deterministic
pattern rules.

---

## Verifying persistence (the 2-minute demo)

**In the UI**

1. Send: `I'm 25, I live in Dhaka and I love football` → the sidebar fills with three facts.
2. **Fully stop** Streamlit (`Ctrl+C`) and start it again.
3. Ask: `What sport should I try this weekend?` → expand *"🧠 memories used"*; `favourite sport: football` is there, and the answer is built around it.
4. Send: `Actually I'm 26 now` → the sidebar shows `age: 26 (was: 25)` — **one** age row, not two.

**From the command line** — each command is a separate OS process, so this is a
genuine restart test:

```bash
python -m memory_chatbot.cli chat "I am 25 years old and I love football"
python -m memory_chatbot.cli chat "What sport should I try?"     # ← recalls football
python -m memory_chatbot.cli chat "Actually I am 26 now"          # ← UPDATE, not ADD
python -m memory_chatbot.cli list
```

Actual output from the third and fourth commands:

```
memory writes:
  update  age: 26 (was: 25)

[health]
  allergy        peanuts               [imp 0.95 | 2026-07-28 18:13 | e476b8a1]
[identity]
  age            26  (was: 25)         [imp 0.90 | 2026-07-28 18:13 | e1d63ac6]
[interest]
  interest       football              [imp 0.55 | 2026-07-28 18:13 | ed41e409]
[location]
  city           Dhaka                 [imp 0.85 | 2026-07-28 18:13 | 32e65e2c]
```

One age row, the old value preserved as `(was: 25)`.

---

## Configuration

All settings come from `.env` (see [`.env.example`](.env.example) for the annotated
full list). The ones that matter:

| Variable | Default | What it controls |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` · `gemini` · `openai` · `echo` (offline) |
| `LLM_MODEL` | *provider default* | Blank → `llama-3.3-70b-versatile` / `gemini-2.0-flash` / `gpt-4o-mini` |
| `MEMORY_BACKEND` | `sqlite` | `sqlite` (built-in) or `mem0` |
| `MEMORY_TOP_K` | `6` | Memories injected per turn |
| `MEMORY_MIN_SCORE` | `0.05` | Relevance floor for recall |
| `MEMORY_EXTRACTION_ENABLED` | `true` | `false` → regex rules only (free, offline) |
| `HISTORY_WINDOW` | `8` | Recent turns replayed verbatim (short-term memory) |
| `EMBEDDING_PROVIDER` | `none` | `none` · `openai` · `gemini` — adds semantic recall |
| `DATA_DIR` | `./data` | Where memories are written |

Configuration is validated by Pydantic at startup, so a bad value fails loudly
with the field name rather than misbehaving later. API keys are `SecretStr` and
never appear in logs or reprs.

---

## Project structure

```
Persistent Memory Chatbot/
├── app.py                      # Streamlit entry point (thin — just wires paths)
├── .env.example                # annotated configuration template
├── requirements.txt            # runtime deps
├── requirements-mem0.txt       # optional: enable the Mem0 backend
├── requirements-dev.txt        # + pytest, ruff
├── pyproject.toml              # packaging, pytest, ruff, coverage config
│
├── src/memory_chatbot/
│   ├── config.py               # env-driven Settings, validated + cached
│   ├── models.py               # MemoryRecord, ChatMessage, categories, operations
│   ├── prompts.py              # every prompt string, in one place
│   ├── logging_utils.py        # idempotent logging setup (Streamlit re-runs!)
│   ├── cli.py                  # memcli — inspect / exercise the store
│   │
│   ├── llm/                    # ── LLMConnector ──────────────────────────
│   │   ├── base.py             #   provider contract + JSON repair
│   │   ├── providers.py        #   Groq · Gemini · OpenAI · offline Echo
│   │   ├── embeddings.py       #   optional embedders + cosine similarity
│   │   └── connector.py        #   prompt assembly, streaming, error handling
│   │
│   ├── memory/                 # ── MemoryHandler ─────────────────────────
│   │   ├── base.py             #   BaseMemoryBackend contract
│   │   ├── sqlite_backend.py   #   default store: SQLite + WAL + slot index
│   │   ├── mem0_backend.py     #   Mem0 OSS store (version-compat shims)
│   │   ├── scoring.py          #   hybrid relevance ranking (pure functions)
│   │   ├── extraction.py       #   FactExtractor + ConflictResolver
│   │   ├── handler.py          #   the policy facade
│   │   └── registry.py         #   backend selection + graceful fallback
│   │
│   ├── chat/                   # ── ChatEngine ────────────────────────────
│   │   ├── engine.py           #   recall → generate → remember
│   │   └── transcript.py       #   durable conversation log
│   │
│   └── ui/                     # ── StreamlitChatUI ───────────────────────
│       ├── streamlit_app.py    #   page flow, session state, cached resources
│       ├── components.py       #   sidebar, memory chips, forms
│       └── styles.py           #   theme-aware CSS
│
├── data/                       # persistent store (git-ignored)
├── docs/
│   ├── memory_tools.md         # market survey — task 1 of the brief
│   └── memory_strategy.md      # what to store / how to recall — task 4
└── tests/                      # 183 tests, no network required
```

The three tools named in the brief map to
[`MemoryHandler`](src/memory_chatbot/memory/handler.py),
[`LLMConnector`](src/memory_chatbot/llm/connector.py) and
[`ui/streamlit_app.py`](src/memory_chatbot/ui/streamlit_app.py).

---

## Architecture

One turn, three steps:

```
                          ┌──────────────────────────┐
   user message ─────────▶│  1. RECALL               │
                          │  MemoryHandler.recall()  │──▶ hybrid ranking
                          └────────────┬─────────────┘    (semantic · lexical ·
                                       │                    importance · recency)
                                       ▼
                          ┌──────────────────────────┐
                          │  2. GENERATE             │
                          │  LLMConnector            │──▶ Groq / Gemini / OpenAI
                          └────────────┬─────────────┘
                                       │  prompt =
                                       │    system: behaviour rules
                                       │    system: MEMORY CONTEXT  ← long-term
                                       │    …last N turns verbatim  ← short-term
                                       │    user:   the question
                                       ▼
                          ┌──────────────────────────┐
   reply ◀────────────────│  3. REMEMBER             │
                          │  extract → reconcile     │──▶ SQLite / Mem0  (disk)
                          └──────────────────────────┘
```

Recall happens **before** generation so the reply is informed; the write happens
**after** so a failed generation can't pollute the store, and the model never sees
a fact it is about to be told.

Design decisions worth calling out:

- **Layers depend inward only.** `ui → chat → {llm, memory} → {models, config}`.
  No layer imports the one above it; `prompts.py` sits at the root because all
  three middle layers need it.
- **Every external dependency degrades.** No API key → offline echo provider.
  Mem0 unavailable → SQLite. Embeddings unavailable → lexical ranking.
  Extraction JSON unparseable → regex rules. The app has no single point of
  failure that takes the memory feature down with it.
- **Errors never cost you a reply.** Provider failures return a readable message;
  a memory-write crash is logged and the reply still renders.
- **Ranking is a pure function of pre-tokenised input**
  ([`scoring.py`](src/memory_chatbot/memory/scoring.py)), which makes recall both
  deterministically testable and cheap to cache.
- **Recall is optimised for the hot path.** Each user's tokens, IDF table and
  unit-normalised vectors are computed once and reused across queries; the
  embedding column is not even read when no embedder is configured; similarity is
  a single `math.sumprod` dot product. Together these took recall over 1 000
  facts from 74 ms to 3 ms (21 ms with live 1536-dim embeddings) — see the table
  in [`docs/memory_strategy.md`](docs/memory_strategy.md#6-known-limits).
- **A write costs one query per category, not one per fact.** A turn that reveals
  four facts reads each category once and keeps the in-memory view in step, so
  two facts from the same turn still see each other.
- **Streamlit's re-run model is respected**: expensive objects live in
  `st.cache_resource` keyed by the settings that built them; logging setup is
  idempotent; SQLite uses WAL with `check_same_thread=False` behind a lock.

---

## Memory strategy summary

Full version: [`docs/memory_strategy.md`](docs/memory_strategy.md).

**What's stored** — one row per fact, keyed by a `(user_id, category, key)` *slot*:

```json
{ "category": "identity", "key": "age", "value": "26",
  "importance": 0.9, "history": ["25"], "updated_at": "2026-07-28T17:36:04Z" }
```

Eleven closed categories: `identity` `location` `preference` `interest`
`occupation` `goal` `health` `relationship` `skill` `constraint` `other`.
A fact is stored only if it is **durable**, **about the user**, and **reusable**.

**How it's recalled** — four blended signals plus a category bridge:

| Signal | Weight |
|---|---|
| Semantic similarity (when embeddings are configured) | 0.45 |
| IDF-weighted lexical overlap | 0.30 |
| Importance | 0.15 |
| Recency (45-day half-life) | 0.10 |
| **+ category bonus** when question keywords match a category | +0.20 |

That bonus is what makes *"What sport should I try?"* recall `football` despite
sharing no words with it.

**How overrides work** — six checks, cheapest first, LLM only at the end:

```
same slot + same value      → NOOP
same slot + new value       → UPDATE   ← "I am 26 now" replaces 25
value already in category   → NOOP     (no double-weighting in recall)
single-valued synonym slot  → UPDATE   (city ↔ location)
empty category              → ADD
otherwise                   → ask the LLM (ADD / UPDATE / DELETE / NOOP)
```

The `UNIQUE(user_id, category, key)` index makes "one live value per attribute"
structurally impossible to violate — the override is enforced by the schema, not
by hoping the model cooperates. Superseded values move to `history`, so nothing
is lost.

---

## Memory backends

### `sqlite` (default) — zero services, zero keys

A single file at `data/memory.db`: WAL journaling, indexed slot lookups, inline
float32 embedding blobs, per-fact history, per-user isolation. Recall ranks one
user's facts in-process, which is fast to tens of thousands of rows and fully
testable.

### `mem0` — the reference memory framework

```bash
pip install -r requirements-mem0.txt
# then in .env:
MEMORY_BACKEND=mem0
OPENAI_API_KEY=...        # or GOOGLE_API_KEY — Mem0 needs an embedder
```

Vectors go to an embedded Qdrant directory and Mem0's own history DB, both under
`data/mem0/`. Our taxonomy travels in Mem0's per-memory `metadata`, so override
semantics are identical on both backends; recall uses Mem0's vector search
re-ranked by our importance/recency scorer. If Mem0 or its embedder is
unavailable, the app logs why and falls back to SQLite rather than failing to start.

> **Honest caveat:** the Mem0 backend targets the 2.x API (verified by inspecting
> `mem0ai` 2.0.14, with fallbacks for 0.1.x) but has **not** been run end-to-end
> here, because that needs an embedding key this environment doesn't have. The
> SQLite path is what the tests and smoke runs cover.

Why SQLite is the default is argued in
[`docs/memory_tools.md#3-what-this-project-uses-and-why`](docs/memory_tools.md).

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                        # 183 tests, no network, no API keys
pytest --cov                  # with coverage
ruff check .                  # lint
```

| File | Tests | Covers |
|---|---|---|
| [`test_extraction.py`](tests/test_extraction.py) | 56 | Rule patterns, LLM parsing, JSON repair, rules-beat-LLM merge, retraction detection |
| [`test_config_and_llm.py`](tests/test_config_and_llm.py) | 45 | Settings validation, secret masking, provider fallback, prompt assembly, model round-trips |
| [`test_sqlite_backend.py`](tests/test_sqlite_backend.py) | 36 | CRUD, slot uniqueness, history capping, ranked recall, corpus-cache invalidation, durability |
| [`test_memory_handler.py`](tests/test_memory_handler.py) | 21 | ADD/UPDATE/DELETE/NOOP policy, synonym slots, resolver safety rails |
| [`test_persistence.py`](tests/test_persistence.py) | 15 | **The brief's acceptance criteria**, across simulated restarts |
| [`test_streamlit_app.py`](tests/test_streamlit_app.py) | 10 | The real page driven through Streamlit's `AppTest`: first render, a full turn, recall on turn two, sidebar contents |

The persistence tests are the important ones. Each builds a `Session` — its own
LLM client, backend and engine — then tears it **all** down and rebuilds from the
same database file, so nothing but the file itself can carry state:

```python
def test_a_new_age_replaces_the_old_one_across_sessions(self, settings):
    with new_session(settings, json_queue=[facts_payload(("identity","age","25",0.9))]) as day1:
        day1.engine.respond(TEST_USER, "I am 25 years old")

    with new_session(settings, json_queue=[facts_payload(("identity","age","26",0.9))]) as day2:
        result = day2.engine.respond(TEST_USER, "I am 26 now")
        assert [w.operation for w in result.writes] == [MemoryOperation.UPDATE]

    with new_session(settings) as day3:                    # ← a third fresh "process"
        records = day3.handler.profile(TEST_USER)
        assert len(records) == 1, "the old age was kept alongside the new one"
        assert records[0].value == "26"
        assert records[0].history == ["25"]
```

The LLM is a `ScriptedLLM` double with queued responses, so tests are
deterministic and free. It records conversation prompts and extraction prompts
separately — a single turn issues both.

---

## CLI

```bash
python -m memory_chatbot.cli info                     # active config + storage paths
python -m memory_chatbot.cli list                     # every stored fact, by category
python -m memory_chatbot.cli recall "what sport"      # what would be injected, with scores
python -m memory_chatbot.cli chat "I am 26 now"       # one full turn
python -m memory_chatbot.cli remember age 27 -c identity
python -m memory_chatbot.cli forget --all
python -m memory_chatbot.cli export -o memories.json

python -m memory_chatbot.cli -u alice list            # a different memory profile
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| *"No LLM API key found… offline echo"* | No key in `.env`. Add `GROQ_API_KEY=…`. Memory still works — this is a deliberate fallback, not a crash. |
| Bot doesn't recall an obvious fact | Check it was stored: `python -m memory_chatbot.cli list`. If it isn't there, extraction skipped it — add it manually via the sidebar, or raise `MEMORY_TOP_K`. |
| Duplicate-looking facts | Two genuinely different slots (e.g. `interest` and `favourite_sport`). Delete one with the sidebar ✕. |
| Memories vanished after restart | `DATA_DIR` changed between runs, or you're in a different working directory. Run `memcli info` to see the resolved path. |
| `MEMORY_BACKEND=mem0` silently uses SQLite | Mem0 or its embedder is missing — check the startup log line. Install `requirements-mem0.txt` and set `OPENAI_API_KEY` or `GOOGLE_API_KEY`. |
| Streamlit reports a stale provider | Provider clients are cached per `(provider, backend, model)`. Restart the app after editing `.env`. |
| IDE flags `streamlit`/`pydantic` as missing | The editor's selected interpreter isn't the one you installed into. Point it at the Python that ran `pip install`. |

---

## License & attribution

Built as a learning project for the Spiral Lab brief. The extract-then-reconcile
memory pipeline is modelled on **Mem0**'s published design; the per-fact history
trail is inspired by **Zep/Graphiti**'s temporal edges; the semantic/episodic
memory split borrows vocabulary from **LangMem**. Credit where it's due — see
[`docs/memory_tools.md`](docs/memory_tools.md).
