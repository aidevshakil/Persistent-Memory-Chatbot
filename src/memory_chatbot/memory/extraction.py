"""Turning conversation turns into durable facts.

Two stages, mirroring how humans remember:

1. :class:`FactExtractor` reads a turn and proposes candidate facts. An LLM does
   the work; a deterministic rule pass catches the highest-value patterns
   (age, name, city, likes) so the feature still works when the model is
   offline, rate-limited, or returns junk JSON.
2. :class:`ConflictResolver` decides whether each candidate is new, replaces a
   stored fact, retracts one, or is already known. The ``(category, key)`` slot
   gives a cheap exact answer for most cases; the LLM is consulted only for the
   genuinely ambiguous ones.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from memory_chatbot.llm.base import BaseLLMClient, LLMError
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.models import (
    CandidateFact,
    MemoryCategory,
    MemoryOperation,
    MemoryRecord,
)
from memory_chatbot.prompts import render_extraction_prompt, render_reconcile_prompt

logger = get_logger(__name__)

MAX_VALUE_LENGTH = 300
MAX_FACTS_PER_TURN = 8

#: Slots that can only ever hold one live value. A new value here always
#: replaces the old one - this is what makes "I am 26 now" override "25"
#: without needing the LLM to agree.
SINGLE_VALUED_KEYS = frozenset(
    {
        "age", "name", "full_name", "nickname", "gender", "birthday", "date_of_birth",
        "city", "country", "location", "timezone", "address", "hometown",
        "job_title", "employer", "company", "occupation", "salary", "industry",
        "marital_status", "partner_name", "relationship_status",
        "height", "weight", "blood_type", "diet", "dietary_restriction",
        "native_language", "phone", "email",
    }
)

#: Categories where a superlative slot is inherently single-valued
#: ("favourite_*", "primary_*", "current_*").
_SINGLE_VALUED_PREFIXES = ("favourite_", "favorite_", "primary_", "current_", "main_", "preferred_")

#: Phrases that mean "forget this", handled before extraction.
_RETRACTION_RE = re.compile(
    r"\b(?:i\s+(?:no\s+longer|don'?t|do\s+not)\s+(?:like|play|live|work|eat|use|watch)"
    r"|(?:i\s+)?(?:quit|stopped|gave\s+up)\b"
    r"|forget\s+(?:that|about|what\s+i\s+said))",
    re.IGNORECASE,
)


def is_single_valued(key: str) -> bool:
    """Whether *key* may hold only one live value at a time."""
    normalised = normalise_key(key)
    return normalised in SINGLE_VALUED_KEYS or normalised.startswith(_SINGLE_VALUED_PREFIXES)


def normalise_key(key: str) -> str:
    """``"Favourite Sport"`` -> ``"favourite_sport"``."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower())
    return cleaned.strip("_")[:60] or "note"


def normalise_value(value: str) -> str:
    """Collapse whitespace, strip first-person phrasing, and cap the length."""
    text = re.sub(r"\s+", " ", str(value).strip())
    text = re.sub(r"^(?:i\s+am|i'?m|my|i\s+have|i\s+like|the\s+user\s+is)\s+", "", text, flags=re.I)
    text = text.rstrip(".!,;")
    return text[:MAX_VALUE_LENGTH]


# ---------------------------------------------------------------------------
# Rule-based fallback extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rule:
    pattern: re.Pattern[str]
    category: MemoryCategory
    key: str
    importance: float
    #: Which regex group holds the value.
    group: int = 1


_RULES: tuple[_Rule, ...] = (
    _Rule(re.compile(r"\bi(?:'m| am|\s+am)\s+(\d{1,3})\s*(?:years?\s*old)?\b", re.I),
          MemoryCategory.IDENTITY, "age", 0.9),
    _Rule(re.compile(r"\bi\s+turned\s+(\d{1,3})\b", re.I),
          MemoryCategory.IDENTITY, "age", 0.9),
    _Rule(re.compile(r"\bmy\s+age\s+is\s+(\d{1,3})\b", re.I),
          MemoryCategory.IDENTITY, "age", 0.9),
    _Rule(re.compile(r"\b(?:my name is|i am called|call me|i'?m)\s+([A-Z][a-zA-Z'-]{1,30})\b"),
          MemoryCategory.IDENTITY, "name", 0.95),
    _Rule(re.compile(r"\bi\s+(?:live|stay|reside)\s+in\s+([A-Za-z][A-Za-z\s,'-]{1,50})", re.I),
          MemoryCategory.LOCATION, "city", 0.85),
    _Rule(re.compile(r"\bi(?:'m| am)\s+from\s+([A-Za-z][A-Za-z\s,'-]{1,50})", re.I),
          MemoryCategory.LOCATION, "hometown", 0.7),
    _Rule(re.compile(r"\bmy\s+favou?rite\s+sport\s+is\s+([A-Za-z][A-Za-z\s-]{1,40})", re.I),
          MemoryCategory.INTEREST, "favourite_sport", 0.7),
    _Rule(re.compile(r"\bmy\s+favou?rite\s+food\s+is\s+([A-Za-z][A-Za-z\s-]{1,40})", re.I),
          MemoryCategory.PREFERENCE, "favourite_food", 0.6),
    _Rule(re.compile(r"\bi\s+(?:work|am\s+working)\s+as\s+(?:an?\s+)?([A-Za-z][A-Za-z\s-]{1,40})", re.I),
          MemoryCategory.OCCUPATION, "job_title", 0.85),
    _Rule(re.compile(r"\bi(?:'m| am)\s+a\s+(software\s+\w+|data\s+\w+|\w+\s+engineer|\w+\s+developer|doctor|teacher|student|designer)\b", re.I),
          MemoryCategory.OCCUPATION, "job_title", 0.8),
    _Rule(re.compile(r"\bi\s+(?:love|like|enjoy|am\s+into)\s+(?:playing\s+|watching\s+)?([A-Za-z][A-Za-z\s-]{1,40})", re.I),
          MemoryCategory.INTEREST, "interest", 0.55),
    _Rule(re.compile(r"\bi\s+(?:hate|dislike|can'?t\s+stand)\s+([A-Za-z][A-Za-z\s-]{1,40})", re.I),
          MemoryCategory.PREFERENCE, "dislike", 0.55),
    _Rule(re.compile(r"\bi(?:'m| am)\s+(?:a\s+)?(vegan|vegetarian|pescatarian)\b", re.I),
          MemoryCategory.HEALTH, "diet", 0.85),
    _Rule(re.compile(r"\bi(?:'m| am)\s+allergic\s+to\s+([A-Za-z][A-Za-z\s,-]{1,40})", re.I),
          MemoryCategory.HEALTH, "allergy", 0.95),
    _Rule(re.compile(r"\bi\s+want\s+to\s+(?:learn|become|start)\s+([A-Za-z][A-Za-z\s-]{1,50})", re.I),
          MemoryCategory.GOAL, "goal", 0.6),
    _Rule(re.compile(r"\bi\s+speak\s+([A-Za-z][A-Za-z\s,-]{1,40})", re.I),
          MemoryCategory.SKILL, "languages", 0.6),
    _Rule(re.compile(r"\bi\s+have\s+a\s+(?:pet\s+)?(dog|cat|bird|fish|rabbit)\b", re.I),
          MemoryCategory.RELATIONSHIP, "pet", 0.6),
)

#: Never store a "value" that is really a filler word.
_VALUE_STOPLIST = frozenset({"it", "that", "this", "them", "you", "there", "here", "so", "a", "an", "the"})

#: A conjunction that starts a new clause.
_CONJUNCTION_SPLIT = re.compile(r"\s+(?:and|but|so|because|which|while|then|although)\b", re.I)

#: A comma or semicolon introducing a new subject - "Dhaka, I love football" is
#: two clauses, while "Dhaka, Bangladesh" is one value, so the boundary only
#: fires when a pronoun or conjunction follows.
_CLAUSE_SPLIT = re.compile(
    r"\s*[,;]\s*(?=(?:i|my|we|he|she|they|it|and|but|also|plus)\b)", re.I
)


def _first_clause(value: str) -> str:
    """Trim a greedy regex match back to its first clause.

    ``"I live in Dhaka, I love football"`` matches the city pattern greedily and
    would otherwise store the whole tail as the city.
    """
    for pattern in (_CLAUSE_SPLIT, _CONJUNCTION_SPLIT):
        value = pattern.split(value, maxsplit=1)[0].strip()
    return value.rstrip(",;")


def rule_based_facts(message: str) -> list[CandidateFact]:
    """Deterministic extraction of high-value patterns.

    Deliberately conservative: a wrong stored fact is far more damaging than a
    missed one, since it silently poisons every later reply.
    """
    facts: dict[tuple[MemoryCategory, str], CandidateFact] = {}
    for rule in _RULES:
        match = rule.pattern.search(message)
        if not match:
            continue
        value = _first_clause(normalise_value(match.group(rule.group)))
        if not value or value.lower() in _VALUE_STOPLIST:
            continue
        facts.setdefault(
            (rule.category, rule.key),
            CandidateFact(
                category=rule.category,
                key=rule.key,
                value=value,
                importance=rule.importance,
                source_message=message,
            ),
        )
    return list(facts.values())


def looks_like_retraction(message: str) -> bool:
    """Whether the user appears to be withdrawing a previously stated fact."""
    return bool(_RETRACTION_RE.search(message))


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


class FactExtractor:
    """Proposes :class:`CandidateFact` objects for a conversation turn."""

    def __init__(self, client: BaseLLMClient, *, use_llm: bool = True) -> None:
        self.client = client
        self.use_llm = use_llm

    def extract(self, user_message: str, assistant_reply: str | None = None) -> list[CandidateFact]:
        """Return the deduplicated union of LLM and rule-based candidates."""
        message = (user_message or "").strip()
        if len(message) < 3:
            return []

        candidates: list[CandidateFact] = []
        if self.use_llm:
            candidates.extend(self._extract_with_llm(message, assistant_reply))
        rules = rule_based_facts(message)

        # Rules win on slots they are confident about: a regex cannot hallucinate
        # an age, but an LLM can round one.
        merged: dict[tuple[str, str], CandidateFact] = {
            (str(f.category), f.key): f for f in candidates
        }
        for fact in rules:
            merged[(str(fact.category), fact.key)] = fact
        return list(merged.values())[:MAX_FACTS_PER_TURN]

    def _extract_with_llm(self, message: str, assistant_reply: str | None) -> list[CandidateFact]:
        try:
            payload = self.client.complete_json(
                render_extraction_prompt(message, assistant_reply), temperature=0.0
            )
        except LLMError as exc:
            logger.warning("LLM extraction unavailable (%s); using rules only", exc)
            return []

        raw_facts = payload.get("facts") if isinstance(payload, dict) else payload
        if not isinstance(raw_facts, list):
            logger.warning("Extractor returned unexpected shape: %r", type(raw_facts).__name__)
            return []

        facts: list[CandidateFact] = []
        for entry in raw_facts[:MAX_FACTS_PER_TURN]:
            if not isinstance(entry, dict):
                continue
            key = normalise_key(entry.get("key", ""))
            value = normalise_value(entry.get("value", ""))
            if not value or not key or value.lower() in _VALUE_STOPLIST:
                continue
            try:
                importance = min(1.0, max(0.0, float(entry.get("importance", 0.5))))
            except (TypeError, ValueError):
                importance = 0.5
            facts.append(
                CandidateFact(
                    category=MemoryCategory.coerce(entry.get("category")),
                    key=key,
                    value=value,
                    importance=importance,
                    source_message=message,
                )
            )
        return facts


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Decision:
    """What to do with one candidate fact."""

    operation: MemoryOperation
    target: MemoryRecord | None = None
    value: str = ""


class ConflictResolver:
    """Decides ADD / UPDATE / DELETE / NOOP for a candidate fact.

    Cheap deterministic checks run first; the LLM is only asked when the answer
    is genuinely ambiguous, which keeps a chat turn to at most one extra call.
    """

    def __init__(self, client: BaseLLMClient, *, use_llm: bool = True) -> None:
        self.client = client
        self.use_llm = use_llm

    def resolve(
        self,
        fact: CandidateFact,
        existing_in_category: Sequence[MemoryRecord],
        *,
        retraction: bool = False,
    ) -> Decision:
        same_slot = next((r for r in existing_in_category if r.key == fact.key), None)

        # 1. Same slot, same value -> already known.
        if same_slot and same_slot.value.casefold() == fact.value.casefold():
            if retraction:
                return Decision(MemoryOperation.DELETE, same_slot, same_slot.value)
            return Decision(MemoryOperation.NOOP, same_slot, same_slot.value)

        # 2. Same slot, new value.
        if same_slot:
            if retraction:
                return Decision(MemoryOperation.DELETE, same_slot, same_slot.value)
            return Decision(MemoryOperation.UPDATE, same_slot, fact.value)

        # 3. A different slot in this category already holds this exact value
        #    ("favourite_sport: football" vs "interest: football"). Storing it
        #    twice would double its weight in every later recall.
        if duplicate := next(
            (r for r in existing_in_category if r.value.casefold() == fact.value.casefold()), None
        ):
            if retraction:
                return Decision(MemoryOperation.DELETE, duplicate, duplicate.value)
            return Decision(MemoryOperation.NOOP, duplicate, duplicate.value)

        # 4. Different slot but a single-valued attribute already holds a value
        #    under a synonym ("city" vs "location") -> replace it.
        if is_single_valued(fact.key):
            twin = next(
                (r for r in existing_in_category if is_single_valued(r.key) and _slots_alias(r.key, fact.key)),
                None,
            )
            if twin:
                return Decision(MemoryOperation.UPDATE, twin, fact.value)

        # 5. Nothing stored in this category -> nothing to conflict with.
        if not existing_in_category:
            return Decision(MemoryOperation.ADD, None, fact.value)

        # 6. Ambiguous: ask the model, but never let a failure lose the fact.
        if self.use_llm and (decision := self._resolve_with_llm(fact, existing_in_category)):
            return decision
        return Decision(MemoryOperation.ADD, None, fact.value)

    def _resolve_with_llm(
        self, fact: CandidateFact, existing: Sequence[MemoryRecord]
    ) -> Decision | None:
        listing = "\n".join(f"- id={r.id} | {r.key}: {r.value}" for r in existing)
        prompt = render_reconcile_prompt(
            new_fact=f"{fact.key}: {fact.value}", category=str(fact.category), existing=listing
        )
        try:
            payload = self.client.complete_json(prompt, temperature=0.0, max_tokens=200)
        except LLMError as exc:
            logger.warning("Conflict resolution unavailable (%s); defaulting to add", exc)
            return None
        if not isinstance(payload, dict):
            return None

        try:
            operation = MemoryOperation(str(payload.get("operation", "add")).lower())
        except ValueError:
            return None
        by_id = {r.id: r for r in existing}
        target = by_id.get(str(payload.get("target_id") or ""))
        value = normalise_value(payload.get("value") or fact.value) or fact.value

        # A model that says "update" without naming a valid target cannot be
        # trusted to have picked one - store the fact rather than clobbering.
        if operation in (MemoryOperation.UPDATE, MemoryOperation.DELETE) and target is None:
            return Decision(MemoryOperation.ADD, None, value)
        return Decision(operation, target, value)


#: Synonymous slot names for single-valued attributes.
_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"city", "location", "hometown", "address"}),
    frozenset({"name", "full_name", "nickname"}),
    frozenset({"job_title", "occupation", "profession"}),
    frozenset({"employer", "company"}),
    frozenset({"diet", "dietary_restriction"}),
    frozenset({"marital_status", "relationship_status"}),
)


def _slots_alias(left: str, right: str) -> bool:
    """Whether two slot names refer to the same single-valued attribute."""
    a, b = normalise_key(left), normalise_key(right)
    if a == b:
        return True
    return any(a in group and b in group for group in _ALIAS_GROUPS)
