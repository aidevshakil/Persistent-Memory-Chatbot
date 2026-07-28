# Memory strategy

Task 4 of the brief: **what to store** and **how to recall it**. This document is
the design; the code that implements it is
[`extraction.py`](../src/memory_chatbot/memory/extraction.py),
[`handler.py`](../src/memory_chatbot/memory/handler.py) and
[`scoring.py`](../src/memory_chatbot/memory/scoring.py).

---

## 1. The shape of a memory

Every stored fact is one row, and the row shape is the whole strategy:

```
(user_id, category, key)  →  value   + importance + timestamps + history
└──────── the slot ─────┘     └── what changes over time ──┘
```

```json
{
  "user_id": "default",
  "category": "identity",
  "key": "age",
  "value": "26",
  "importance": 0.9,
  "created_at": "2026-07-27T09:14:22Z",
  "updated_at": "2026-07-28T17:36:04Z",
  "history": ["25"]
}
```

**Why a `(category, key)` slot instead of free-text memories?** Because the brief
requires that *"I am 26 now" replaces 25*. If memories are opaque sentences, the
only way to detect that collision is to ask an LLM and hope. With a slot, the
collision is a database fact: `UNIQUE(user_id, category, key)`. The override is
guaranteed by the schema, and the LLM is left to do the part it is actually good
at — reading language.

`history` keeps superseded values. Deleting the old value outright would lose the
fact that it *was* true, which is exactly the audit trail Zep's temporal graph
provides — obtained here for the cost of one JSON column.

---

## 2. What to store

### The taxonomy

Extraction is constrained to a **closed vocabulary** of 11 categories. A closed
set makes recall predictable (a category can be boosted when the question is
clearly about it) and makes collisions detectable.

| Category | Holds | Example |
|---|---|---|
| `identity` | name, age, gender, birthday | `age: 26` |
| `location` | city, country, timezone | `city: Dhaka` |
| `preference` | likes, dislikes, favourites | `favourite_food: biryani` |
| `interest` | hobbies, sports, topics followed | `favourite_sport: football` |
| `occupation` | job, employer, studies | `job_title: data analyst` |
| `goal` | objectives, plans, targets | `fitness_goal: run a marathon` |
| `health` | diet, allergies, conditions | `allergy: peanuts` |
| `relationship` | partner, family, friends, pets | `pet: cat` |
| `skill` | languages, tools, proficiency | `languages: Bangla, English` |
| `constraint` | budget, schedule, hard limits | `budget: tight` |
| `other` | anything durable that fits nowhere else | |

### The three tests a candidate fact must pass

1. **Durable** — still true next week. `"I'm 26"` yes; `"I'm tired right now"` no.
2. **About the user** — not about the assistant, and not a question. `"What sport
   should I try?"` stores nothing.
3. **Reusable** — would plausibly improve a future answer.

### Importance, and why it exists

`0.0`–`1.0`, used as a ranking signal and as a tie-breaker:

| Range | Meaning | Examples |
|---|---|---|
| `0.85–1.0` | Identity and hard constraints — wrong answers here are embarrassing or unsafe | age, name, allergy |
| `0.5–0.8` | Stable preferences, goals, occupation | favourite sport, job |
| `0.2–0.5` | Passing remarks | "chess is fun" |

### Two extractors, not one

| | LLM extractor | Rule extractor |
|---|---|---|
| Coverage | Broad — paraphrase, implication, compound sentences | ~17 high-value patterns |
| Cost | One JSON call per turn | Free |
| Works offline | No | Yes |
| Can hallucinate | Yes | No |

They run together and their results merge, **with the rules winning any slot they
both claim**. The reasoning: a regex cannot invent an age, but a model can round
one — observed in testing, where a model returned `"twenty five"` for `"I am 25"`.
On the slots where a regex is confident, it is simply more trustworthy.

This also means memory extraction **degrades rather than dies**: no API key, a
rate limit, or unparseable JSON leaves the rule pass still working. That is why
the offline `echo` provider can demonstrate full persistence.

### What is deliberately *not* stored

- Assistant messages (they'd create a feedback loop of the bot remembering itself)
- Questions, small talk, transient state
- Raw conversation as "memory" — the transcript is stored separately, for display
  only, and never enters the recall ranking

---

## 3. How to recall

### Retrieval is not the same as search

The naive approach — embed the question, return the top-k nearest facts — fails
on the brief's own example. *"What sport should I try this weekend?"* has near-zero
lexical overlap with `favourite_sport: football`, and even embeddings rank it
below whatever fact happens to mention "weekend".

So recall blends **four signals**:

| Signal | Weight | Why it's needed |
|---|---|---|
| **Semantic** | 0.45 | Paraphrase matching — only active when an embedding provider is configured |
| **Lexical** | 0.30 | IDF-weighted term overlap. A user whose every fact mentions "football" gets little signal from that word; a rare token like "vegan" is highly discriminative |
| **Importance** | 0.15 | An allergy should surface more readily than a passing remark |
| **Recency** | 0.10 | Exponential decay, ~45-day half-life. Recently confirmed facts win ties |

Weights are **renormalised over the signals actually available**, so switching
embeddings off doesn't silently shrink every score below the threshold.

### Plus a category bridge

On top of the blend, a **+0.20 category bonus** fires when question keywords map
to a category: *"sport"*, *"play"*, *"weekend"*, *"try"* → `interest`; *"recipe"*,
*"eat"*, *"diet"* → `health`; *"where"*, *"weather"*, *"local"* → `location`.

This is what makes the brief's scenario work. Verified end-to-end:

```
$ python -m memory_chatbot.cli chat "What sport should I try?"
recalled:
  0.532  interest: football     ← ranked first, despite zero word overlap
  0.427  age: 25
```

### Then: injection

The top `MEMORY_TOP_K` (default 6) facts above `MEMORY_MIN_SCORE` are grouped by
category and injected as a **system** message, ahead of the conversation:

```
1. system  — behaviour rules ("use memory naturally, never invent it")
2. system  — MEMORY CONTEXT, grouped by category      ← long-term memory
3. …       — the last HISTORY_WINDOW turns, verbatim  ← short-term memory
4. user    — the live question
```

Order is deliberate: nothing outranks what the user just said. If a stored fact
contradicts the live message, the prompt instructs the model to trust the user —
and the extractor then writes the correction back.

**Two memory horizons, not one.** The recent-turns window handles "it" and "that
one" within a conversation; the ranked long-term store handles what survives a
restart. Conflating them is the usual reason chatbots either forget mid-sentence
or drown in stale context.

---

## 4. Reconciliation: how an override actually happens

When a candidate fact arrives, six checks run **in order**, cheapest first. The
LLM is consulted only at step 6, so a normal turn costs at most one extra call:

| # | Condition | Result |
|---|---|---|
| 1 | Same slot, same value | `NOOP` |
| 2 | Same slot, **different** value | **`UPDATE`** ← *age 25 → 26* |
| 3 | Same value already stored under a different key in this category | `NOOP` (prevents double-weighting the same fact in recall) |
| 4 | Single-valued attribute already filled under a synonym (`city` ↔ `location`) | **`UPDATE`** |
| 5 | Category empty — nothing to conflict with | `ADD` |
| 6 | Genuinely ambiguous | Ask the LLM → `ADD` / `UPDATE` / `DELETE` / `NOOP` |

### Single-valued vs multi-valued

Some attributes can only hold one live value; others legitimately hold several.
A person has one age but may enjoy several sports. `SINGLE_VALUED_KEYS` lists the
former explicitly, plus any key prefixed `favourite_`, `primary_`, `current_`,
`main_`, `preferred_` — because a "favourite" is single-valued by definition.

### Retractions

`"I no longer play football"`, `"forget what I said about X"`, `"I quit my job"` are
detected before reconciliation and turn the matched slot into a `DELETE`.

### Two safety rails

- If the LLM says `UPDATE`/`DELETE` but names no valid target row, the decision
  degrades to `ADD`. A model that cannot identify what it wants to overwrite is
  not trusted to overwrite anything.
- If the reconciliation call fails entirely, the fact is `ADD`ed. Storing a
  near-duplicate is recoverable; silently dropping what the user just told you
  is not.

### Importance never downgrades

An update keeps `max(old, new)`. Learning your age a second time in passing
shouldn't make it less important than when you first stated it.

---

## 5. Worked example — the brief's scenario, verbatim

**Day 1** — `"I like football"`

```
extract    → interest / favourite_sport / "football" / 0.7
reconcile  → category empty                          → ADD
disk       → data/memory.db
```

**App closed. Process exits. Later:**

**Day 2** — `"What sport should I try?"`

```
recall     → "sport"/"try" → interest bonus +0.20 → football scores 0.532, ranked 1st
inject     → MEMORY CONTEXT: [interest] favourite sport: football
reply      → "Given how much you enjoy football, try a five-a-side league…"
extract    → no durable facts in a question           → nothing written
```

**Day 3** — `"Actually I'm 26 now"` (age 25 stored previously)

```
extract    → identity / age / "26" / 0.9   (rule pass, exact)
reconcile  → same slot, different value                → UPDATE
result     → value "26", history ["25"], still exactly one age row
```

Every step above is covered by a test in
[`test_persistence.py`](../tests/test_persistence.py), each simulating a real
restart by discarding all in-process state and reopening the database file.

---

## 6. Known limits

Honest about what this design does *not* do:

- **No relational reasoning.** Flat facts, no graph. "My sister's husband works
  where I used to" is stored as text, not as traversable edges. Zep/Graphiti is
  the right tool if that matters.
- **No temporal queries.** `history` records *that* a value changed, not the
  interval each value was valid for. "Where did I live last March?" is unanswerable.
- **Extraction is single-turn.** A fact spread across three turns ("I'm moving" …
  "to Chittagong") may be missed; only the previous assistant reply is given as
  disambiguating context.
- **Ranking is in-process.** All of one user's facts are scored in Python rather
  than by a vector index. Three things keep that cheap: the embedding column is
  only read when an embedder is configured, each user's tokens/IDF/normalised
  vectors are cached and reused across queries, and similarity is a single
  pre-normalised dot product via `math.sumprod`. Measured on this machine:

  | Facts for one user | Recall, no embeddings | Recall, 1536-dim embeddings |
  |---|---|---|
  | 100 | 0.3 ms | 2.3 ms |
  | 1 000 | 3.0 ms | 21 ms |
  | 5 000 | 17 ms | — |

  The cache is rebuilt after a write, so the first query of a turn costs an extra
  ~14 ms per 100 embedded facts. Beyond roughly 10 000 facts per user a real
  vector index becomes the right answer.
- **Trust is unconditional.** Anything stated is believed. There is no
  corroboration threshold before a fact is written.
