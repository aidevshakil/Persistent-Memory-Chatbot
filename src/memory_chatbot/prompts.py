"""All prompt text lives here so tuning behaviour never means touching logic.

Deliberately at the package root: the llm, memory and chat layers all need
prompts, and putting them inside any one of those would create an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

from memory_chatbot.models import MemoryCategory, MemorySearchResult

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a warm, concise personal assistant with a long-term memory of this user.

How to use memory:
- A MEMORY CONTEXT block may be supplied below. Treat it as verified facts the \
user told you in earlier sessions.
- Use those facts naturally, the way a friend who remembers would. Do not \
announce that you are "checking your memory" or list the raw facts back.
- When a memory makes an answer more personal, lean on it. If the user asks \
"what sport should I try" and you remember they like football, build the answer \
around that.
- If the memory context is empty or irrelevant to the question, just answer \
normally. Never invent a memory you were not given.
- If a stored fact contradicts what the user says right now, trust the user and \
move on without arguing about it.

Style: friendly, direct, 1-3 short paragraphs unless more detail is asked for.\
"""

MEMORY_CONTEXT_HEADER = "MEMORY CONTEXT - long-term facts recalled for this turn:"

NO_MEMORY_NOTE = "MEMORY CONTEXT: (empty - nothing stored about this user yet)"


def render_memory_context(results: Sequence[MemorySearchResult]) -> str:
    """Render recalled memories as a system block, grouped by category.

    Grouping helps the model tell "who the user is" apart from "what they like",
    which measurably reduces mixing unrelated facts into one sentence.
    """
    if not results:
        return NO_MEMORY_NOTE

    by_category: dict[MemoryCategory, list[MemorySearchResult]] = {}
    for result in results:
        by_category.setdefault(result.record.category, []).append(result)

    lines = [MEMORY_CONTEXT_HEADER]
    for category, items in by_category.items():
        lines.append(f"[{category}]")
        lines.extend(f"- {item.record.as_sentence()}" for item in items)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

_CATEGORY_LIST = ", ".join(str(c) for c in MemoryCategory)

EXTRACTION_SYSTEM_PROMPT = f"""\
You extract durable, long-term facts about the user from a conversation turn.

Return STRICT JSON only:
{{"facts": [{{"category": "...", "key": "...", "value": "...", "importance": 0.0}}]}}

Rules:
1. category MUST be one of: {_CATEGORY_LIST}
2. key is a short snake_case slot name that will be REUSED when this fact \
changes later. Use stable, generic slots: "age", "name", "city", \
"favourite_sport", "job_title", "dietary_restriction", "fitness_goal". \
Never put the value inside the key (use key "favourite_sport" / value \
"football", NOT key "likes_football").
3. value is a short third-person statement of the fact, no pronouns like "I". \
Good: "26". "Dhaka, Bangladesh". "football". Bad: "I am 26 years old".
4. importance is 0.0-1.0: 0.9 for identity and hard constraints, 0.6 for \
stable preferences and goals, 0.3 for passing remarks.
5. Extract ONLY things that stay true beyond this conversation. Skip questions, \
opinions about the assistant, small talk, transient state ("I'm tired right \
now"), and anything the assistant said about itself.
6. One fact per object. Split compound statements into separate facts.
7. If the turn contains no durable facts, return {{"facts": []}}. Do not guess.

Examples
Input: "I'm 25 and I live in Dhaka. I love football."
Output: {{"facts": [
  {{"category": "identity", "key": "age", "value": "25", "importance": 0.9}},
  {{"category": "location", "key": "city", "value": "Dhaka", "importance": 0.8}},
  {{"category": "interest", "key": "favourite_sport", "value": "football", "importance": 0.6}}
]}}

Input: "What sport should I try this weekend?"
Output: {{"facts": []}}\
"""

EXTRACTION_USER_TEMPLATE = """\
Conversation turn to analyse.

User said:
\"\"\"{user_message}\"\"\"

{assistant_context}
Extract the durable facts about the user as JSON."""


def render_extraction_prompt(user_message: str, assistant_reply: str | None = None) -> list[dict]:
    """Build the messages for one extraction call."""
    assistant_context = ""
    if assistant_reply:
        # Included only for disambiguation (e.g. the user answers "yes, 26").
        assistant_context = (
            "Assistant had just said (context only - do not extract facts "
            f'about the assistant):\n"""{assistant_reply[:600]}"""\n'
        )
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACTION_USER_TEMPLATE.format(
                user_message=user_message, assistant_context=assistant_context
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

RECONCILE_SYSTEM_PROMPT = """\
You maintain a user's long-term memory store. Given a NEW fact and the \
EXISTING facts stored in the same category, decide what to do.

Return STRICT JSON only:
{"operation": "add" | "update" | "delete" | "noop", "target_id": "... or null", \
"value": "final value to store"}

Decision rules:
- "update" + target_id: the new fact replaces an existing one because it is the \
same attribute with a changed or more precise value (age 25 -> 26, city Dhaka \
-> Chittagong, "likes football" -> "now prefers cricket"). This is the most \
common case for anything that can change over time.
- "delete" + target_id: the user explicitly retracted the existing fact \
("I don't play football anymore" with nothing replacing it).
- "noop": an existing fact already says this. Prefer noop over adding a \
near-duplicate.
- "add": genuinely new information that does not touch any existing fact.

Two facts can coexist only if they are truly independent (a person may like \
several sports). A single-valued attribute - age, name, city, job, a "favourite" \
- must never have two live values: update instead.\
"""

RECONCILE_USER_TEMPLATE = """\
NEW fact:
{new_fact}

EXISTING facts in category "{category}":
{existing}

Decide the operation as JSON."""


def render_reconcile_prompt(new_fact: str, category: str, existing: str) -> list[dict]:
    """Build the messages for one conflict-resolution call."""
    return [
        {"role": "system", "content": RECONCILE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RECONCILE_USER_TEMPLATE.format(
                new_fact=new_fact, category=category, existing=existing or "(none)"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Session summary (shown in the sidebar)
# ---------------------------------------------------------------------------

PROFILE_SUMMARY_PROMPT = """\
Summarise who this user is in at most three short sentences, using only the \
facts listed. No preamble, no bullet points, no invented details.

Facts:
{facts}"""
