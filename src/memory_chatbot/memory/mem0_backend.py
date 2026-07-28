"""Mem0 (OSS) memory backend.

Enabled with ``MEMORY_BACKEND=mem0`` after ``pip install -r requirements-mem0.txt``.

Mem0 stores vectors in an embedded Qdrant directory plus its own SQLite history
db, both under ``data/mem0/``, so persistence across restarts is preserved.

Two deliberate choices:

* Our structured taxonomy (``category``/``key``/``importance``) travels in Mem0's
  per-memory ``metadata``, so :class:`~memory_chatbot.memory.handler.MemoryHandler`
  keeps identical override semantics on either backend.
* Recall uses Mem0's own vector search, then re-ranks with our scorer so the
  category/importance/recency signals still apply.

Mem0's API changed between 0.1.x and 2.x - reads moved from a top-level
``user_id=`` argument to ``filters={"user_id": ...}``, and ``update(data=…)``
became ``update(text=…)``. Every call therefore goes through the ``_call_*``
compatibility shims, and every read through :func:`_iter_results`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from memory_chatbot.config import LLMProvider, Settings
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.memory.base import HISTORY_LIMIT, BaseMemoryBackend, MemoryBackendError
from memory_chatbot.memory.scoring import build_idf, document_tokens, score, tokenize
from memory_chatbot.models import (
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
    utc_now,
)

logger = get_logger(__name__)

#: Mem0 LLM provider ids keyed by ours (Mem0 says "gemini", not "google").
_MEM0_LLM_PROVIDER = {
    LLMProvider.GROQ: "groq",
    LLMProvider.GEMINI: "gemini",
    LLMProvider.OPENAI: "openai",
}

#: Upper bound for "give me everything for this user" reads.
_MAX_FETCH = 1000


def _iter_results(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield memory dicts from any Mem0 response shape (list or ``{"results": …}``)."""
    if payload is None:
        return
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("memories", payload.get("data", [])))
    if isinstance(payload, dict):
        payload = [payload]
    for item in payload or []:
        if isinstance(item, dict):
            yield item


class Mem0MemoryBackend(BaseMemoryBackend):
    """:class:`BaseMemoryBackend` implementation on top of ``mem0.Memory``."""

    name = "mem0"

    def __init__(self, settings: Settings, *, memory: Any | None = None) -> None:
        self.settings = settings
        self.root = Path(settings.mem0_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.location = str(self.root)
        # Mem0 has no indexed slot lookup, so find_slot/by_category have to scan
        # a full fetch. Caching it turns a multi-fact turn from N fetches into 1.
        self._cache: dict[str, list[MemoryRecord]] = {}

        if memory is not None:  # injected in tests
            self._memory = memory
            return
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise MemoryBackendError(
                "MEMORY_BACKEND=mem0 requires the mem0ai package: "
                "pip install -r requirements-mem0.txt"
            ) from exc
        try:
            self._memory = Memory.from_config(self.build_config())
        except Exception as exc:
            raise MemoryBackendError(
                f"Could not initialise Mem0: {exc}. Mem0 needs an embedding "
                "provider - set OPENAI_API_KEY or GOOGLE_API_KEY, or use "
                "MEMORY_BACKEND=sqlite."
            ) from exc
        logger.info("Mem0 backend ready at %s", self.root)

    # ------------------------------------------------------------------ config
    def build_config(self) -> dict[str, Any]:
        """Build a fully local Mem0 config from our settings."""
        settings = self.settings
        provider = settings.llm_provider
        if provider not in _MEM0_LLM_PROVIDER:
            provider = LLMProvider.OPENAI

        config: dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memory_chatbot",
                    "path": str(self.root / "qdrant"),
                    "on_disk": True,
                },
            },
            "history_db_path": str(self.root / "history.db"),
        }
        if api_key := settings.api_key_for(provider):
            config["llm"] = {
                "provider": _MEM0_LLM_PROVIDER[provider],
                "config": {
                    "model": settings.llm_model,
                    "temperature": 0.1,
                    "api_key": api_key,
                },
            }
        # Mem0 always needs an embedder; only OpenAI/Gemini keys can serve one.
        if settings.openai_api_key:
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": settings.openai_api_key.get_secret_value(),
                },
            }
        elif settings.google_api_key:
            config["embedder"] = {
                "provider": "gemini",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": settings.google_api_key.get_secret_value(),
                },
            }
        return config

    # -------------------------------------------------------- version shims
    def _call_get_all(self, user_id: str) -> Any:
        """``get_all`` across Mem0 versions (filters= in 2.x, user_id= in 0.1.x)."""
        try:
            return self._memory.get_all(filters={"user_id": user_id}, top_k=_MAX_FETCH)
        except (TypeError, ValueError):
            return self._memory.get_all(user_id=user_id)

    def _call_search(self, query: str, user_id: str, limit: int) -> Any:
        """``search`` across Mem0 versions; threshold 0.0 so our scorer decides."""
        try:
            return self._memory.search(
                query, filters={"user_id": user_id}, top_k=limit, threshold=0.0
            )
        except (TypeError, ValueError):
            return self._memory.search(query, user_id=user_id, limit=limit)

    def _call_update(self, memory_id: str, text: str) -> Any:
        """``update`` across Mem0 versions (``text=`` in 2.x, ``data=`` before)."""
        try:
            return self._memory.update(memory_id=memory_id, text=text)
        except TypeError:
            return self._memory.update(memory_id=memory_id, data=text)

    # ------------------------------------------------------------------- write
    def add(self, record: MemoryRecord) -> MemoryRecord:
        if existing := self.find_slot(record.user_id, str(record.category), record.key):
            return self.update(
                existing.id,
                value=record.value,
                importance=record.importance,
                source_message=record.source_message,
            )
        try:
            response = self._memory.add(
                record.as_sentence(),
                user_id=record.user_id,
                metadata=self._metadata(record),
                infer=False,  # our handler already extracted and reconciled
            )
        except Exception as exc:
            raise MemoryBackendError(f"Mem0 add failed: {exc}") from exc

        for item in _iter_results(response):
            if item_id := item.get("id"):
                record.id = str(item_id)
                break
        self._cache.pop(record.user_id, None)
        logger.info("mem0 add | %s | %s", record.user_id, record.as_sentence())
        return record

    def update(
        self,
        record_id: str,
        *,
        value: str,
        importance: float | None = None,
        embedding: list[float] | None = None,
        source_message: str | None = None,
    ) -> MemoryRecord:
        current = self.get(record_id)
        if current is None:
            raise MemoryBackendError(f"Cannot update unknown Mem0 memory {record_id}")
        if value != current.value:
            current.history = [current.value, *current.history][:HISTORY_LIMIT]
        current.value = value
        current.importance = current.importance if importance is None else importance
        current.source_message = source_message or current.source_message
        current.updated_at = utc_now()
        try:
            self._call_update(record_id, current.as_sentence())
        except Exception as exc:
            raise MemoryBackendError(f"Mem0 update failed: {exc}") from exc
        self._cache.pop(current.user_id, None)
        logger.info("mem0 update | %s | %s", current.user_id, current.as_sentence())
        return current

    def delete(self, record_id: str) -> bool:
        try:
            self._memory.delete(memory_id=record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem0 delete failed for %s: %s", record_id, exc)
            return False
        self._cache.clear()  # the deleted row's owner is no longer knowable
        return True

    def clear(self, user_id: str) -> int:
        count = self.count(user_id)
        try:
            self._memory.delete_all(user_id=user_id)
        except Exception as exc:
            raise MemoryBackendError(f"Mem0 delete_all failed: {exc}") from exc
        self._cache.pop(user_id, None)
        return count

    # -------------------------------------------------------------------- read
    def get(self, record_id: str) -> MemoryRecord | None:
        try:
            payload = self._memory.get(memory_id=record_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem0 get failed for %s: %s", record_id, exc)
            return None
        for item in _iter_results(payload):
            return self._to_record(item)
        return None

    def all(self, user_id: str) -> list[MemoryRecord]:
        if (cached := self._cache.get(user_id)) is not None:
            return list(cached)
        try:
            payload = self._call_get_all(user_id)
        except Exception as exc:
            raise MemoryBackendError(f"Mem0 get_all failed: {exc}") from exc
        records = [self._to_record(item) for item in _iter_results(payload)]
        records.sort(key=lambda r: r.updated_at, reverse=True)
        self._cache[user_id] = records
        return list(records)

    def find_slot(self, user_id: str, category: str, key: str) -> MemoryRecord | None:
        return next(
            (
                r
                for r in self.all(user_id)
                if str(r.category) == str(category) and r.key == key
            ),
            None,
        )

    def by_category(self, user_id: str, category: str) -> list[MemoryRecord]:
        return [r for r in self.all(user_id) if str(r.category) == str(category)]

    def search(
        self, user_id: str, query: str, *, limit: int = 6, min_score: float = 0.0
    ) -> list[MemorySearchResult]:
        """Mem0 vector search, re-ranked with our category/importance/recency scorer."""
        try:
            payload = self._call_search(query, user_id, max(limit * 3, 10))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mem0 search failed (%s); falling back to get_all", exc)
            payload = None

        items = list(_iter_results(payload)) if payload is not None else []
        if items:
            records: Sequence[MemoryRecord] = [self._to_record(item) for item in items]
            vector_scores = {
                str(item.get("id")): float(item.get("score") or 0.0) for item in items
            }
        else:
            records = self.all(user_id)
            vector_scores = {}
        if not records:
            return []

        query_tokens = tokenize(query)
        doc_tokens = [document_tokens(record) for record in records]
        idf = build_idf(doc_tokens)
        now = utc_now()

        results: list[MemorySearchResult] = []
        for record, tokens in zip(records, doc_tokens, strict=True):
            local = score(
                record, doc_tokens=tokens, query_tokens=query_tokens, idf=idf, now=now
            )
            vector = max(0.0, min(1.0, vector_scores.get(record.id, 0.0)))
            # Mem0's vector score leads when present; ours breaks ties.
            results.append(
                MemorySearchResult(
                    record=record, score=0.6 * vector + 0.4 * local if vector else local
                )
            )

        results.sort(key=lambda item: (-item.score, -item.record.importance))
        return [item for item in results if item.score >= min_score][:limit]

    def users(self) -> list[str]:
        # Mem0 OSS has no "list users" API; the UI falls back to the active user.
        return []

    # ----------------------------------------------------------------- mapping
    @staticmethod
    def _metadata(record: MemoryRecord) -> dict[str, Any]:
        return {
            "category": str(record.category),
            "key": record.key,
            "value": record.value,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "source_message": record.source_message,
            "history": record.history,
        }

    @staticmethod
    def _to_record(item: dict[str, Any]) -> MemoryRecord:
        """Map a Mem0 memory dict onto :class:`MemoryRecord`.

        Memories Mem0 inferred on its own carry no metadata, so the text is
        parsed as ``"key: value"`` with sensible fallbacks.
        """
        metadata = item.get("metadata") or {}
        text = str(item.get("memory") or item.get("text") or metadata.get("value") or "")
        key = metadata.get("key")
        value = metadata.get("value")
        if not key or not value:
            head, _, tail = text.partition(":")
            if tail:
                key = key or head.strip().replace(" ", "_").lower()
                value = value or tail.strip()
            else:
                key = key or "note"
                value = value or text
        return MemoryRecord.from_dict(
            {
                "id": str(item.get("id") or ""),
                "user_id": str(item.get("user_id") or metadata.get("user_id") or "default"),
                "category": MemoryCategory.coerce(metadata.get("category")),
                "key": key,
                "value": value,
                "importance": metadata.get("importance", 0.5),
                "created_at": metadata.get("created_at") or item.get("created_at"),
                "updated_at": (
                    metadata.get("updated_at") or item.get("updated_at") or item.get("created_at")
                ),
                "source_message": metadata.get("source_message"),
                "history": metadata.get("history") or [],
            }
        )
