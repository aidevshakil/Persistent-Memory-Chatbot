"""Storage contract every memory backend implements.

The handler above this layer owns *policy* (what to store, when to override);
a backend owns only *persistence and retrieval*. That split is what lets Mem0
and the built-in SQLite store be swapped with one env var.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memory_chatbot.models import MemoryRecord, MemorySearchResult

#: How many superseded values to keep per fact.
HISTORY_LIMIT = 10


class MemoryBackendError(RuntimeError):
    """Raised when the underlying store cannot satisfy a request.

    Backends must translate driver-specific failures into this type; the layers
    above only handle it, so a raw driver exception would break a chat turn.
    """


class BaseMemoryBackend(ABC):
    """Persistent store for :class:`MemoryRecord` objects."""

    #: Identifier shown in the UI.
    name: str = "backend"
    #: Where the data physically lives (shown in the UI).
    location: str = ""

    # ------------------------------------------------------------------ write
    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert *record*, or update in place if its slot is already taken."""

    @abstractmethod
    def update(
        self,
        record_id: str,
        *,
        value: str,
        importance: float | None = None,
        embedding: list[float] | None = None,
        source_message: str | None = None,
    ) -> MemoryRecord:
        """Replace the value of an existing record, archiving the old value."""

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Remove a record. Returns whether anything was deleted."""

    @abstractmethod
    def clear(self, user_id: str) -> int:
        """Delete every memory for *user_id*; returns the number removed."""

    # ------------------------------------------------------------------- read
    @abstractmethod
    def get(self, record_id: str) -> MemoryRecord | None:
        """Fetch one record by id."""

    @abstractmethod
    def all(self, user_id: str) -> list[MemoryRecord]:
        """Every memory for *user_id*, newest update first.

        Embeddings are not guaranteed to be populated - they are an internal
        ranking detail, and loading them is expensive.
        """

    @abstractmethod
    def find_slot(self, user_id: str, category: str, key: str) -> MemoryRecord | None:
        """Exact ``(user, category, key)`` lookup - the fast path for overrides."""

    @abstractmethod
    def by_category(self, user_id: str, category: str) -> list[MemoryRecord]:
        """Live facts in one category - the conflict-resolution context."""

    @abstractmethod
    def search(
        self, user_id: str, query: str, *, limit: int = 6, min_score: float = 0.0
    ) -> list[MemorySearchResult]:
        """Relevance-ranked recall for *query*."""

    # ----------------------------------------------------------- housekeeping
    def users(self) -> list[str]:
        """All user ids known to the store; empty if the store cannot list them."""
        return []

    def count(self, user_id: str) -> int:
        return len(self.all(user_id))

    def close(self) -> None:  # noqa: B027 - optional hook; not every store holds resources
        """Release resources. Safe to call more than once."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} location={self.location!r}>"
