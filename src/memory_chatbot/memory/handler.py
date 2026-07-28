"""``MemoryHandler`` - stores and retrieves persistent facts about the user.

The only memory API the rest of the app touches. It owns the *policy* (what is
worth storing, when a new fact overrides an old one, how much to recall) and
delegates *persistence* to a :class:`~memory_chatbot.memory.base.BaseMemoryBackend`.
"""

from __future__ import annotations

from memory_chatbot.config import Settings, get_settings
from memory_chatbot.llm.base import BaseLLMClient
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.memory.base import BaseMemoryBackend, MemoryBackendError
from memory_chatbot.memory.extraction import (
    ConflictResolver,
    FactExtractor,
    looks_like_retraction,
    normalise_key,
    normalise_value,
)
from memory_chatbot.models import (
    CandidateFact,
    MemoryCategory,
    MemoryOperation,
    MemoryRecord,
    MemorySearchResult,
    MemoryWrite,
)

logger = get_logger(__name__)


class MemoryHandler:
    """Facade over extraction, conflict resolution and a persistence backend."""

    def __init__(
        self,
        backend: BaseMemoryBackend,
        llm_client: BaseLLMClient,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        use_llm = self.settings.memory_extraction_enabled
        self.backend = backend
        self.extractor = FactExtractor(llm_client, use_llm=use_llm)
        self.resolver = ConflictResolver(llm_client, use_llm=use_llm)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @property
    def storage_location(self) -> str:
        return self.backend.location

    # ------------------------------------------------------------------ recall
    def recall(
        self,
        user_id: str,
        query: str,
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve the memories most relevant to *query*.

        Returns an empty list rather than raising: a recall failure should
        degrade the reply, never break the conversation.
        """
        try:
            return self.backend.search(
                user_id,
                query,
                limit=self.settings.memory_top_k if limit is None else limit,
                min_score=self.settings.memory_min_score if min_score is None else min_score,
            )
        except MemoryBackendError as exc:
            logger.error("Recall failed for %s: %s", user_id, exc)
            return []

    def profile(self, user_id: str) -> list[MemoryRecord]:
        """Every stored fact for *user_id*, newest first."""
        try:
            return self.backend.all(user_id)
        except MemoryBackendError as exc:
            logger.error("Could not load profile for %s: %s", user_id, exc)
            return []

    def profile_by_category(self, user_id: str) -> dict[MemoryCategory, list[MemoryRecord]]:
        """Stored facts grouped by category, for the sidebar and the CLI."""
        grouped: dict[MemoryCategory, list[MemoryRecord]] = {}
        for record in self.profile(user_id):
            grouped.setdefault(record.category, []).append(record)
        return dict(sorted(grouped.items(), key=lambda kv: str(kv[0])))

    def count(self, user_id: str) -> int:
        try:
            return self.backend.count(user_id)
        except MemoryBackendError:
            return 0

    # ------------------------------------------------------------------ ingest
    def remember_turn(
        self, user_id: str, user_message: str, assistant_reply: str | None = None
    ) -> list[MemoryWrite]:
        """Extract facts from a turn and reconcile them into the store."""
        facts = self.extractor.extract(user_message, assistant_reply)
        if not facts:
            return []
        retraction = looks_like_retraction(user_message)

        # One read per distinct category instead of one per fact, kept in step
        # with each write so two facts from the same turn see each other.
        context: dict[str, list[MemoryRecord]] = {}
        writes = []
        for fact in facts:
            category = str(fact.category)
            if category not in context:
                context[category] = self._in_category(user_id, category)
            write = self._persist(user_id, fact, context[category], retraction=retraction)
            _apply_to_context(context[category], write)
            writes.append(write)
        return writes

    def remember_fact(
        self,
        user_id: str,
        *,
        category: MemoryCategory | str,
        key: str,
        value: str,
        importance: float = 0.7,
    ) -> MemoryWrite:
        """Store a fact directly - the manual "add memory" UI and the CLI."""
        fact = CandidateFact(
            category=MemoryCategory.coerce(category),
            key=normalise_key(key),
            value=normalise_value(value),
            importance=importance,
        )
        existing = self._in_category(user_id, str(fact.category))
        return self._persist(user_id, fact, existing, retraction=False)

    def _persist(
        self,
        user_id: str,
        fact: CandidateFact,
        existing: list[MemoryRecord],
        *,
        retraction: bool,
    ) -> MemoryWrite:
        """Apply one candidate fact to the store and describe what happened."""
        decision = self.resolver.resolve(fact, existing, retraction=retraction)
        try:
            match decision.operation:
                case MemoryOperation.UPDATE if decision.target is not None:
                    previous = decision.target.value
                    record = self.backend.update(
                        decision.target.id,
                        value=decision.value or fact.value,
                        importance=max(decision.target.importance, fact.importance),
                        source_message=fact.source_message,
                    )
                    return MemoryWrite(MemoryOperation.UPDATE, record, previous)

                case MemoryOperation.DELETE if decision.target is not None:
                    self.backend.delete(decision.target.id)
                    return MemoryWrite(
                        MemoryOperation.DELETE, decision.target, decision.target.value
                    )

                case MemoryOperation.NOOP if decision.target is not None:
                    return MemoryWrite(MemoryOperation.NOOP, decision.target, decision.target.value)

                case _:
                    record = self.backend.add(
                        MemoryRecord(
                            user_id=user_id,
                            category=fact.category,
                            key=fact.key,
                            value=decision.value or fact.value,
                            importance=fact.importance,
                            source_message=fact.source_message,
                        )
                    )
                    return MemoryWrite(MemoryOperation.ADD, record)
        except MemoryBackendError as exc:
            logger.error("Could not persist %s=%s: %s", fact.key, fact.value, exc)
            placeholder = MemoryRecord(
                user_id=user_id, category=fact.category, key=fact.key, value=fact.value
            )
            return MemoryWrite(MemoryOperation.NOOP, placeholder)

    def _in_category(self, user_id: str, category: str) -> list[MemoryRecord]:
        try:
            return list(self.backend.by_category(user_id, category))
        except MemoryBackendError as exc:
            logger.error("Could not read category %s for %s: %s", category, user_id, exc)
            return []

    # ------------------------------------------------------------ maintenance
    def forget(self, record_id: str) -> bool:
        try:
            return self.backend.delete(record_id)
        except MemoryBackendError as exc:
            logger.error("Delete failed for %s: %s", record_id, exc)
            return False

    def forget_all(self, user_id: str) -> int:
        try:
            return self.backend.clear(user_id)
        except MemoryBackendError as exc:
            logger.error("Clear failed for %s: %s", user_id, exc)
            return 0

    def export(self, user_id: str) -> list[dict]:
        """JSON-serialisable dump of a user's memories."""
        return [record.to_dict() for record in self.profile(user_id)]

    def known_users(self) -> list[str]:
        try:
            return self.backend.users()
        except MemoryBackendError:
            return []

    def close(self) -> None:
        self.backend.close()


def _apply_to_context(records: list[MemoryRecord], write: MemoryWrite) -> None:
    """Mirror a completed write into the in-memory category view."""
    match write.operation:
        case MemoryOperation.ADD:
            records.append(write.record)
        case MemoryOperation.DELETE:
            records[:] = [r for r in records if r.id != write.record.id]
        case MemoryOperation.UPDATE:
            for index, record in enumerate(records):
                if record.id == write.record.id:
                    records[index] = write.record
                    break
