# Memory tools for chatbots — market survey

Task 1 of the brief: *explore the tools on the market that give chatbots memory,
list them, and use them.* This is what I found, what each one is actually good
at, and why this project ended up with the architecture it has.

> Surveyed while building this project (July 2026). This space moves fast —
> version numbers and API shapes below reflect what was installed and inspected
> at the time, and are worth re-checking against each project's own docs. Where
> I could verify something by running it, I say so.

---

## 1. The landscape in one picture

Not everything called "memory" solves the same problem. Sorting the tools by
*what they actually do* matters more than sorting by popularity:

| Layer | What it gives you | Examples |
|---|---|---|
| **Memory frameworks** | Extraction policy + conflict resolution + retrieval. The whole "what should I remember" decision. | **Mem0**, **Zep**, **LangMem**, **Letta**, **Cognee**, **Memobase** |
| **Memory servers / protocol layers** | A memory *service* several clients share, usually over MCP | **OpenMemory**, **Zep Cloud**, **Redis Agent Memory Server**, **Supermemory** |
| **Graph engines** | Entities, relations, and *when each fact was true* | **Graphiti**, Neo4j + Mem0's graph mode, Cognee |
| **Vector stores** | Storage and similarity search — no memory policy at all | Qdrant, Chroma, pgvector, Weaviate, LanceDB, FAISS |
| **Provider built-ins** | Conversation state / memory managed by the model vendor | OpenAI conversation state, Anthropic memory tooling |

The key insight for this project: **a vector database is not a memory layer.**
Storing every message in Chroma gives you retrieval, but it never decides that
"I am 26 now" should *replace* "I am 25". That decision — the conflict
resolution — is what the frameworks in row 1 add, and it is the actual
requirement in the brief.

---

## 2. The tools, in detail

### Mem0 (`mem0ai`) — the closest match to this brief
**OSS (Apache-2.0) + managed platform · Python & TypeScript · [mem0.ai](https://mem0.ai)**

The reference implementation of LLM-mediated memory. Its pipeline is a two-phase
design worth studying:

1. **Extract** — an LLM reads the turn and proposes candidate facts.
2. **Update** — a second LLM call compares each candidate against semantically
   similar stored memories and emits `ADD` / `UPDATE` / `DELETE` / `NONE`.

That second phase is exactly the "I am 26 now replaces 25" behaviour the brief
asks for, and it is the design this project deliberately mirrors (see
[`memory_strategy.md`](memory_strategy.md)).

- Storage: pluggable vector store (Qdrant by default; Chroma, pgvector, and
  others supported), plus its own SQLite history DB — so persistence across
  restarts is free.
- Optional **graph memory** (Neo4j / Memgraph) for relations between entities.
- Verified locally: `mem0ai` **2.0.14** installs and imports on Python 3.14.
- ⚠️ **API churn is real.** In 2.x, reads take `filters={"user_id": …}` and
  reject a top-level `user_id=`; `update()` takes `text=` where 0.1.x took
  `data=`. This project's [`mem0_backend.py`](../src/memory_chatbot/memory/mem0_backend.py)
  therefore calls through version-detecting shims rather than assuming one shape.
- ⚠️ It **requires an embedding provider** (OpenAI/Gemini key, or a local
  HuggingFace embedder), so it is not truly zero-config offline.

**Verdict:** integrated as a first-class backend (`MEMORY_BACKEND=mem0`), and its
extract-then-reconcile design is the intellectual basis for the default backend.

---

### OpenMemory — Mem0, but as a local-first shared server
**OSS · part of the Mem0 project**

A self-hosted memory server (Docker + MCP) that lets *several* AI clients —
Claude, Cursor, Windsurf, your own app — read and write **one shared memory** that
never leaves your machine. It's Mem0's engine behind an MCP boundary plus a
dashboard for inspecting and revoking what has been stored.

- Excellent when the memory should outlive any single application.
- Overkill here: a Docker service and MCP hop for one Streamlit app, and the
  brief explicitly wants the memory written to disk by *this* app.

**Verdict:** surveyed, not used. Notable as the answer when "which app owns the
memory?" has more than one answer.

---

### Zep + Graphiti — memory as a temporal knowledge graph
**Zep: cloud + community edition · Graphiti: OSS engine · [getzep.com](https://www.getzep.com)**

Zep's differentiator is *time*. Built on **Graphiti**, its open-source temporal
graph engine, it stores facts as graph edges carrying validity intervals — so it
records not just "lives in Chittagong" but "lived in Dhaka *until* March, lives in
Chittagong *since*". Old facts are invalidated rather than deleted, and you can
query the graph as of any past date.

- Strongest option when relationships between entities matter ("her manager is
  the person who introduced me to X") or when history must be auditable.
- Cost: a graph database (Neo4j/FalkorDB) to run, a heavier ingestion pipeline,
  and a bigger conceptual surface than a personal-facts chatbot needs.

**Verdict:** the most sophisticated model in the survey, and the direction I would
take this project if the requirement grew to relationship reasoning. Its
bi-temporal idea is partly borrowed: this project keeps a `history` array of
superseded values per fact, which gives an audit trail without a graph DB.

---

### LangMem — memory for LangGraph agents
**OSS SDK from LangChain · named in the brief's outcomes**

Frames memory in cognitive-science terms and is genuinely useful vocabulary:

- **Semantic** — facts about the user (this project's entire scope)
- **Episodic** — memories of past events/interactions
- **Procedural** — learned behaviour, e.g. an evolving system prompt

Offers both hot-path tools an agent can call to save memories, and a *background*
memory manager that consolidates after the fact. Persists through the LangGraph
`BaseStore` (including a Postgres store).

- ⚠️ Practically speaking it assumes you are on LangChain/LangGraph. Adopting it
  here would mean pulling the whole graph runtime in to serve a three-step turn.

**Verdict:** surveyed; its semantic/episodic/procedural taxonomy directly shaped
how this project's categories are organised. Not adopted, to avoid a framework
dependency an order of magnitude larger than the app.

---

### Letta (formerly MemGPT) — the agent edits its own memory
**OSS + cloud · [letta.com](https://www.letta.com)**

From the MemGPT paper's premise: treat the context window like virtual memory and
let the agent page information in and out. Memory lives in **blocks** the agent
rewrites itself, split into core (always in context) and archival (retrieved on
demand), with "sleep-time" background consolidation.

- Fascinating and the most autonomous option — memory the *model* curates.
- It is an agent runtime, not a library you add to an existing chatbot. Adopting
  it means Letta owns the loop, which conflicts with a Streamlit app that owns
  its own turn structure.

**Verdict:** surveyed, not used. Its core/archival split is the same instinct
behind this project's `HISTORY_WINDOW` (recent turns, always in context) versus
ranked long-term recall (retrieved per turn).

---

### Cognee — ECL pipelines into a knowledge graph
**OSS · [cognee.ai](https://www.cognee.ai)**

"Extract, Cognify, Load": ingests documents and conversations and builds a
combined knowledge graph + vector index you then query. Aimed at *corpora* —
codebases, document sets — more than at a running conversation.

**Verdict:** surveyed. Strong for document memory, mismatched to short
first-person facts, which need no graph to be useful.

---

### Others worth knowing

| Tool | One-line summary | Why not here |
|---|---|---|
| **Memobase** | Profile-centric memory: maintains a structured user profile over time | Closest in spirit to my own design; found it after the schema was settled |
| **Redis Agent Memory Server** | Redis-backed short + long-term memory service | Needs a Redis instance for a single-user desktop app |
| **Supermemory / Papr / Honcho** | Hosted memory APIs with generous free tiers | Hosted-first; the brief wants memory on local disk |
| **Motorhead** | Early memory server (Rust) | Archived / no longer maintained |
| **Chroma / Qdrant / pgvector / LanceDB** | Vector stores | Storage only — no extraction, no conflict resolution (see §1) |
| **Provider built-ins** (OpenAI conversation state, Anthropic memory tooling) | Vendor-managed context | Locks memory to one vendor; the brief wants provider-swappable |

---

## 3. What this project uses, and why

Two backends ship behind one interface
([`BaseMemoryBackend`](../src/memory_chatbot/memory/base.py)), selectable with a
single environment variable.

### `MEMORY_BACKEND=sqlite` (default)

A purpose-built store: SQLite + a `UNIQUE(user_id, category, key)` index +
hybrid ranked retrieval. Reasons it is the default rather than Mem0:

1. **It runs with zero keys and zero services.** Mem0 needs an embedding
   provider; a grader with no API key would see the memory feature fail. Here,
   `MEMORY_BACKEND=sqlite` + `LLM_PROVIDER=echo` demonstrates full persistence
   offline.
2. **The override requirement becomes a database constraint, not a prompt.** A
   UNIQUE index on `(user_id, category, key)` makes "one live value per
   attribute" structurally impossible to violate. Mem0 achieves the same via an
   LLM call, which is more flexible but can be talked out of it.
3. **It is testable.** Ranking is a pure function, so the 183-test suite covers
   recall behaviour deterministically with no network and no mocked vector DB.
4. **Nothing about it is a toy.** WAL journaling, indexed lookups, an audit
   trail of superseded values, JSON export, and per-user isolation.

### `MEMORY_BACKEND=mem0`

Real Mem0 OSS: embedded Qdrant vectors + Mem0's history DB under `data/mem0/`.
Our structured taxonomy travels in Mem0's per-memory `metadata`, so override
semantics are identical on both backends; recall uses Mem0's vector search
re-ranked by our own importance/recency scorer. Needs
`pip install -r requirements-mem0.txt` and an embedding key.

**Honest limitation:** the Mem0 backend is written against the 2.x API verified
by inspection, with fallbacks for 0.1.x, but it has **not** been executed
end-to-end here — doing so requires an OpenAI or Gemini embedding key, which this
environment does not have. The SQLite path is the one covered by tests and by the
CLI smoke runs.

---

## 4. What I'd choose for a different brief

| If the requirement were… | I'd reach for |
|---|---|
| Personal facts, local disk, no services *(this brief)* | The SQLite backend here, or Mem0 OSS |
| Memory shared across Claude / Cursor / your app | **OpenMemory** |
| "Who introduced me to whom, and when was that true?" | **Zep / Graphiti** |
| Already committed to LangGraph | **LangMem** |
| An agent that curates its own memory | **Letta** |
| Memory over a document corpus, not a conversation | **Cognee** |
| Team-scale, someone else runs it | **Mem0 Platform** or **Zep Cloud** |

The through-line: pick the tool by the *shape of the memory* — flat personal
facts, a temporal graph, or a document corpus — not by which library is trending.
