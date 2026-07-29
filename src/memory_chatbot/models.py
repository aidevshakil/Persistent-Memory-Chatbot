"""Domain objects shared across the memory, LLM, chat and UI layers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self


def utc_now() -> datetime:
    """Timezone-aware "now"; every timestamp in the app goes through here."""
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Role(StrEnum):
    """Chat message author."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MemoryCategory(StrEnum):
    """Taxonomy the extractor is constrained to.

    A closed vocabulary keeps recall predictable (we can boost a category when
    the question is about it) and makes conflict detection reliable, because
    "I am 25" and "I am 26 now" land in the same ``(category, key)`` slot.
    """

    IDENTITY = "identity"          # name, age, gender, family
    LOCATION = "location"          # city, country, timezone
    PREFERENCE = "preference"      # likes, dislikes, favourites
    INTEREST = "interest"          # hobbies, sports, topics followed
    OCCUPATION = "occupation"      # job, employer, studies
    GOAL = "goal"                  # objectives, plans, targets
    HEALTH = "health"              # diet, allergies, conditions
    RELATIONSHIP = "relationship"  # partner, friends, pets
    SKILL = "skill"                # languages, tools, proficiency
    CONSTRAINT = "constraint"      # budget, schedule, hard limits
    OTHER = "other"

    @classmethod
    def coerce(cls, value: object) -> MemoryCategory:
        """Best-effort parse of an LLM-produced category string."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        return cls.OTHER


class MemoryOperation(StrEnum):
    """Outcome of reconciling a candidate fact against stored memory."""

    ADD = "add"        # genuinely new fact
    UPDATE = "update"  # supersedes an existing fact (age 25 -> 26)
    DELETE = "delete"  # user retracted a fact
    NOOP = "noop"      # already known, nothing to write


@dataclass(slots=True)
class ChatMessage:
    """One turn of the conversation."""

    role: Role
    content: str
    created_at: datetime = field(default_factory=utc_now)
    id: str = field(default_factory=new_id)
    #: Memory ids injected into the prompt for this turn (for the UI "why").
    used_memory_ids: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, str]:
        return {"role": str(self.role), "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": str(self.role),
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "used_memory_ids": list(self.used_memory_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            role=Role(payload["role"]),
            content=payload["content"],
            created_at=_parse_dt(payload.get("created_at")),
            id=payload.get("id") or new_id(),
            used_memory_ids=list(payload.get("used_memory_ids") or []),
        )


@dataclass(slots=True)
class MemoryRecord:
    """A single persisted fact about the user.

    ``key`` is the normalised slot name (``"age"``, ``"favourite_sport"``) and is
    what makes updates possible: a new value for a known ``(user_id, category,
    key)`` triple replaces the old one instead of piling up beside it.
    """

    user_id: str
    category: MemoryCategory
    key: str
    value: str
    importance: float = 0.5
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    source_message: str | None = None
    embedding: list[float] | None = None
    #: Superseded values, newest first - keeps an audit trail of overrides.
    history: list[str] = field(default_factory=list)

    def as_sentence(self) -> str:
        """Human/LLM readable rendering used when injecting memory context."""
        label = self.key.replace("_", " ")
        return f"{label}: {self.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "category": str(self.category),
            "key": self.key,
            "value": self.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_message": self.source_message,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        return cls(
            user_id=payload["user_id"],
            category=MemoryCategory.coerce(payload.get("category")),
            key=payload["key"],
            value=payload["value"],
            importance=float(payload.get("importance", 0.5)),
            id=payload.get("id") or new_id(),
            created_at=_parse_dt(payload.get("created_at")),
            updated_at=_parse_dt(payload.get("updated_at")),
            source_message=payload.get("source_message"),
            embedding=payload.get("embedding"),
            history=list(payload.get("history") or []),
        )


@dataclass(slots=True)
class CandidateFact:
    """A fact proposed by the extractor, before reconciliation."""

    category: MemoryCategory
    key: str
    value: str
    importance: float = 0.5
    source_message: str | None = None


@dataclass(slots=True)
class MemorySearchResult:
    """A recalled memory together with its relevance score."""

    record: MemoryRecord
    score: float

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))


@dataclass(slots=True)
class MemoryWrite:
    """What the handler did with one candidate fact - surfaced in the UI."""

    operation: MemoryOperation
    record: MemoryRecord
    previous_value: str | None = None

    def describe(self) -> str:
        match self.operation:
            case MemoryOperation.ADD:
                return f"+ {self.record.as_sentence()}"
            case MemoryOperation.UPDATE:
                return f"~ {self.record.as_sentence()} (was: {self.previous_value})"
            case MemoryOperation.DELETE:
                return f"- {self.record.as_sentence()}"
            case _:
                return f"= {self.record.as_sentence()}"


@dataclass(slots=True)
class ChatTurnResult:
    """Everything the UI needs to render one completed exchange."""

    reply: ChatMessage
    recalled: list[MemorySearchResult] = field(default_factory=list)
    writes: list[MemoryWrite] = field(default_factory=list)

    @property
    def persisted_writes(self) -> list[MemoryWrite]:
        return [w for w in self.writes if w.operation is not MemoryOperation.NOOP]


def _parse_dt(value: object) -> datetime:
    """Parse an ISO timestamp, defaulting to now and always UTC-aware."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return utc_now()
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return utc_now()
