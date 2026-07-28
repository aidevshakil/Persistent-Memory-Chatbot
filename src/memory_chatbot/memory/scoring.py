"""Relevance scoring for memory recall.

Recall blends four signals so the app works with *or* without an embedding
provider:

``lexical``    IDF-weighted term overlap between the query and the fact
``semantic``   cosine similarity of embeddings (skipped when unavailable)
``importance`` how load-bearing the fact is (identity > passing remark)
``recency``    freshly confirmed facts win ties against stale ones

Weights are renormalised over whichever signals are available, so turning
embeddings off never silently shrinks every score below the threshold.

Every function here is pure and takes *pre-tokenised* input, which lets the
backend cache tokenisation and keeps per-query work to set intersections.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime

from memory_chatbot.llm.embeddings import unit_similarity
from memory_chatbot.models import MemoryCategory, MemoryRecord

WEIGHT_SEMANTIC = 0.45
WEIGHT_LEXICAL = 0.30
WEIGHT_IMPORTANCE = 0.15
WEIGHT_RECENCY = 0.10

#: Half-life of the recency bonus, in days.
RECENCY_HALFLIFE_DAYS = 45.0

CATEGORY_HINT_BONUS = 0.20

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    he her him his how i if in into is it its me my of on or our should so some such
    than that the their them then there these they this to too was we were what when
    where which who why will with would you your about just get got very really am
    tell me know remember think want need like""".split()
)

#: Question keywords that imply a category, so a fact can be recalled even when
#: it shares no words with the question ("what sport?" -> favourite_sport).
_CATEGORY_HINTS: dict[MemoryCategory, frozenset[str]] = {
    MemoryCategory.IDENTITY: frozenset({"age", "old", "name", "born", "birthday", "call"}),
    MemoryCategory.LOCATION: frozenset({"live", "city", "country", "where", "nearby", "local", "weather"}),
    MemoryCategory.PREFERENCE: frozenset({"favourite", "favorite", "prefer", "love", "hate", "like", "best"}),
    MemoryCategory.INTEREST: frozenset({"sport", "hobby", "play", "watch", "read", "music", "game", "try", "fun", "weekend"}),
    MemoryCategory.OCCUPATION: frozenset({"work", "job", "career", "study", "company", "office", "salary"}),
    MemoryCategory.GOAL: frozenset({"goal", "plan", "want", "achieve", "target", "improve", "learn"}),
    MemoryCategory.HEALTH: frozenset({"eat", "food", "diet", "allergy", "health", "gym", "fit", "sleep", "recipe", "meal"}),
    MemoryCategory.RELATIONSHIP: frozenset({"wife", "husband", "partner", "family", "friend", "pet", "dog", "cat", "kids"}),
    MemoryCategory.SKILL: frozenset({"skill", "language", "code", "speak", "good", "learn", "experience"}),
    MemoryCategory.CONSTRAINT: frozenset({"budget", "cheap", "afford", "time", "busy", "cannot", "avoid", "limit"}),
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords and single characters removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def document_tokens(record: MemoryRecord) -> frozenset[str]:
    """The searchable token set for one record."""
    return frozenset(tokenize(f"{record.key.replace('_', ' ')} {record.value} {record.category}"))


def build_idf(documents: Sequence[Collection[str]]) -> dict[str, float]:
    """Inverse document frequency across one user's own memories.

    A user whose every fact mentions "football" gets little signal from that
    word, while a rare token like "vegan" becomes highly discriminative.
    """
    total = len(documents) or 1
    doc_freq: dict[str, int] = {}
    for tokens in documents:
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {
        token: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }


def lexical_score(
    query_tokens: Collection[str], doc_tokens: Collection[str], idf: Mapping[str, float]
) -> float:
    """IDF-weighted overlap of query and document tokens, in ``[0, 1]``."""
    if not query_tokens or not doc_tokens:
        return 0.0
    unique = set(query_tokens)
    ideal = sum(idf.get(t, 1.0) for t in unique)
    if not ideal:
        return 0.0
    matched = sum(idf.get(t, 1.0) for t in unique if t in doc_tokens)
    return min(1.0, matched / ideal)


def recency_score(updated_at: datetime, *, now: datetime) -> float:
    """Exponential decay with a 45-day half-life."""
    age_days = max(0.0, (now - updated_at).total_seconds() / 86_400.0)
    return math.exp(-age_days / RECENCY_HALFLIFE_DAYS)


def category_bonus(query_tokens: Collection[str], category: MemoryCategory) -> float:
    """Flat boost when the question is clearly about *category*."""
    hints = _CATEGORY_HINTS.get(category)
    if hints and not hints.isdisjoint(query_tokens):
        return CATEGORY_HINT_BONUS
    return 0.0


_BASE_WEIGHT = WEIGHT_LEXICAL + WEIGHT_IMPORTANCE + WEIGHT_RECENCY


def score(
    record: MemoryRecord,
    *,
    doc_tokens: Collection[str],
    query_tokens: Collection[str],
    idf: Mapping[str, float],
    now: datetime,
    doc_vector: Sequence[float] | None = None,
    query_vector: Sequence[float] | None = None,
) -> float:
    """Combined relevance of one record for one query, in ``[0, 1]``.

    ``doc_vector`` and ``query_vector`` must already be unit-normalised; the
    caller normalises each once rather than per comparison.
    """
    weighted = (
        WEIGHT_LEXICAL * lexical_score(query_tokens, doc_tokens, idf)
        + WEIGHT_IMPORTANCE * min(1.0, max(0.0, record.importance))
        + WEIGHT_RECENCY * recency_score(record.updated_at, now=now)
    )
    total_weight = _BASE_WEIGHT
    if doc_vector and query_vector:
        weighted += WEIGHT_SEMANTIC * unit_similarity(query_vector, doc_vector)
        total_weight += WEIGHT_SEMANTIC

    combined = weighted / total_weight + category_bonus(query_tokens, record.category)
    return max(0.0, min(1.0, combined))
