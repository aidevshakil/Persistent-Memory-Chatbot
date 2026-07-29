"""Mem0 backend: response-shape handling, CRUD and the override round-trip.

The real ``mem0.Memory`` needs an embedding provider and a Qdrant directory, so
these tests drive the backend through fakes that reproduce Mem0's *response
shapes* exactly - that is where the backend's bugs live, not in Mem0 itself:

* ``get_all``/``search`` answer ``{"results": [...]}``
* ``get`` answers a **bare** memory dict, with ``id``/``memory`` at the top level
* ``update`` in 2.x takes ``text=``/``metadata=``; 0.1.x took only ``data=``

:class:`FakeMemory` mirrors 2.0.x; :class:`LegacyFakeMemory` mirrors 0.1.x, so
the compatibility shims are exercised rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memory_chatbot.config import Settings
from memory_chatbot.memory.base import MemoryBackendError
from memory_chatbot.memory.handler import MemoryHandler
from memory_chatbot.memory.mem0_backend import Mem0MemoryBackend, _iter_results
from memory_chatbot.models import MemoryCategory, MemoryOperation, MemoryRecord

from .conftest import TEST_USER, ScriptedLLM, facts_payload


class FakeMemory:
    """Stand-in for ``mem0.Memory`` 2.x, matching its response shapes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self._next = 0

    # -- write ------------------------------------------------------------
    def add(
        self,
        messages: str,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        self._next += 1
        memory_id = f"mem-{self._next}"
        self.rows[memory_id] = {
            "id": memory_id,
            "memory": messages,
            "user_id": user_id,
            "metadata": dict(metadata or {}),
        }
        return {"results": [{"id": memory_id, "memory": messages, "event": "ADD"}]}

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        row = self.rows[memory_id]
        if text is not None:
            row["memory"] = text
        if metadata is not None:
            row["metadata"] = dict(metadata)
        return {"message": "Memory updated!"}

    def delete(self, memory_id: str) -> dict[str, Any]:
        self.rows.pop(memory_id, None)
        return {"message": "Memory deleted!"}

    def delete_all(self, user_id: str | None = None, **_: Any) -> dict[str, Any]:
        for key, row in list(self.rows.items()):
            if row["user_id"] == user_id:
                del self.rows[key]
        return {"message": "Memories deleted!"}

    # -- read -------------------------------------------------------------
    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Mem0 returns a bare memory dict here - no ``results`` wrapper."""
        row = self.rows.get(memory_id)
        return dict(row) if row else None

    def get_all(
        self, *, filters: dict[str, Any] | None = None, top_k: int = 20, **_: Any
    ) -> dict[str, Any]:
        user_id = (filters or {}).get("user_id")
        rows = [dict(r) for r in self.rows.values() if r["user_id"] == user_id]
        return {"results": rows[:top_k]}

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        **_: Any,
    ) -> dict[str, Any]:
        user_id = (filters or {}).get("user_id")
        rows = [
            {**row, "score": 0.9 if query.lower() in str(row["memory"]).lower() else 0.1}
            for row in self.rows.values()
            if row["user_id"] == user_id
        ]
        return {"results": rows[:top_k]}


class LegacyFakeMemory(FakeMemory):
    """Mem0 0.1.x: keyword-only ``user_id`` reads and ``data=`` writes."""

    def get_all(self, *, user_id: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        if "filters" in kwargs:
            raise TypeError("get_all() got an unexpected keyword argument 'filters'")
        return [dict(r) for r in self.rows.values() if r["user_id"] == user_id]

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        if "filters" in kwargs:
            raise TypeError("search() got an unexpected keyword argument 'filters'")
        user_id = kwargs.get("user_id")
        return [dict(r) for r in self.rows.values() if r["user_id"] == user_id]

    def update(self, memory_id: str, data: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if "text" in kwargs or "metadata" in kwargs:
            raise TypeError("update() got an unexpected keyword argument")
        self.rows[memory_id]["memory"] = data
        return {"message": "Memory updated!"}


@pytest.fixture
def mem0_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def fake() -> FakeMemory:
    return FakeMemory()


@pytest.fixture
def backend(mem0_settings: Settings, fake: FakeMemory) -> Mem0MemoryBackend:
    return Mem0MemoryBackend(mem0_settings, memory=fake)


def make_record(key: str, value: str, **kwargs: Any) -> MemoryRecord:
    return MemoryRecord(
        user_id=kwargs.pop("user_id", TEST_USER),
        category=kwargs.pop("category", MemoryCategory.IDENTITY),
        key=key,
        value=value,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Response-shape handling
# ---------------------------------------------------------------------------


class TestIterResults:
    def test_unwraps_a_results_container(self) -> None:
        assert list(_iter_results({"results": [{"id": "a"}, {"id": "b"}]})) == [
            {"id": "a"},
            {"id": "b"},
        ]

    def test_accepts_a_plain_list(self) -> None:
        assert list(_iter_results([{"id": "a"}])) == [{"id": "a"}]

    def test_yields_a_bare_memory_dict(self) -> None:
        """``Memory.get`` returns one memory with no container key around it."""
        bare = {"id": "a", "memory": "age: 25", "metadata": {"key": "age"}}
        assert list(_iter_results(bare)) == [bare]

    def test_none_yields_nothing(self) -> None:
        assert list(_iter_results(None)) == []

    def test_skips_non_dict_entries(self) -> None:
        assert list(_iter_results(["nonsense", {"id": "a"}])) == [{"id": "a"}]


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCrud:
    def test_add_then_get_round_trips(self, backend: Mem0MemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        loaded = backend.get(stored.id)
        assert loaded is not None
        assert (loaded.key, loaded.value) == ("age", "25")

    def test_get_unknown_id_returns_none(self, backend: Mem0MemoryBackend) -> None:
        assert backend.get("does-not-exist") is None

    def test_add_to_a_taken_slot_updates_instead_of_duplicating(
        self, backend: Mem0MemoryBackend
    ) -> None:
        backend.add(make_record("age", "25"))
        backend.add(make_record("age", "26"))
        assert [r.value for r in backend.all(TEST_USER)] == ["26"]

    def test_update_of_unknown_id_raises(self, backend: Mem0MemoryBackend) -> None:
        with pytest.raises(MemoryBackendError):
            backend.update("nope", value="x")

    def test_delete_removes_the_record(self, backend: Mem0MemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        assert backend.delete(stored.id) is True
        assert backend.count(TEST_USER) == 0

    def test_clear_only_touches_one_user(self, backend: Mem0MemoryBackend) -> None:
        backend.add(make_record("age", "25"))
        backend.add(make_record("age", "40", user_id="someone-else"))
        assert backend.clear(TEST_USER) == 1
        assert backend.count(TEST_USER) == 0
        assert backend.count("someone-else") == 1


# ---------------------------------------------------------------------------
# The override - the behaviour the whole project is built around
# ---------------------------------------------------------------------------


class TestOverride:
    def test_new_value_is_read_back_not_the_old_one(self, backend: Mem0MemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="26")
        assert backend.find_slot(TEST_USER, "identity", "age").value == "26"

    def test_superseded_value_is_kept_as_history(self, backend: Mem0MemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="26")
        assert backend.find_slot(TEST_USER, "identity", "age").history == ["25"]

    def test_update_writes_the_metadata_mirror_too(
        self, backend: Mem0MemoryBackend, fake: FakeMemory
    ) -> None:
        """Text and metadata must not drift apart; reads trust the metadata."""
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="26")
        row = fake.rows[stored.id]
        assert row["memory"] == "age: 26"
        assert row["metadata"]["value"] == "26"

    def test_importance_survives_an_update(self, backend: Mem0MemoryBackend) -> None:
        stored = backend.add(make_record("age", "25", importance=0.5))
        backend.update(stored.id, value="26", importance=0.9)
        assert backend.find_slot(TEST_USER, "identity", "age").importance == 0.9

    def test_repeated_updates_stack_history_newest_first(
        self, backend: Mem0MemoryBackend
    ) -> None:
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="26")
        backend.update(stored.id, value="27")
        record = backend.find_slot(TEST_USER, "identity", "age")
        assert (record.value, record.history) == ("27", ["26", "25"])


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_by_category_filters(self, backend: Mem0MemoryBackend) -> None:
        backend.add(make_record("age", "25"))
        backend.add(make_record("city", "Dhaka", category=MemoryCategory.LOCATION))
        assert [r.key for r in backend.by_category(TEST_USER, "location")] == ["city"]

    def test_find_slot_misses_on_a_different_category(
        self, backend: Mem0MemoryBackend
    ) -> None:
        backend.add(make_record("age", "25"))
        assert backend.find_slot(TEST_USER, "location", "age") is None

    def test_all_isolates_users(self, backend: Mem0MemoryBackend) -> None:
        backend.add(make_record("age", "25"))
        backend.add(make_record("age", "40", user_id="other"))
        assert [r.value for r in backend.all(TEST_USER)] == ["25"]

    def test_search_ranks_the_matching_memory_first(self, backend: Mem0MemoryBackend) -> None:
        backend.add(make_record("age", "25"))
        backend.add(
            make_record("favourite_sport", "football", category=MemoryCategory.INTEREST)
        )
        results = backend.search(TEST_USER, "football")
        assert results[0].record.value == "football"

    def test_search_honours_the_limit(self, backend: Mem0MemoryBackend) -> None:
        for index in range(5):
            backend.add(make_record(f"key_{index}", f"value_{index}"))
        assert len(backend.search(TEST_USER, "value", limit=2)) == 2

    def test_search_on_an_empty_profile_returns_nothing(
        self, backend: Mem0MemoryBackend
    ) -> None:
        assert backend.search(TEST_USER, "anything") == []


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class TestToRecord:
    def test_parses_an_inferred_memory_that_has_no_metadata(self) -> None:
        record = Mem0MemoryBackend._to_record(
            {"id": "x", "memory": "favourite sport: cricket", "user_id": "u"}
        )
        assert (record.key, record.value) == ("favourite_sport", "cricket")

    def test_falls_back_to_a_note_when_the_text_has_no_slot(self) -> None:
        record = Mem0MemoryBackend._to_record({"id": "x", "memory": "Loves long walks"})
        assert (record.key, record.value) == ("note", "Loves long walks")

    def test_metadata_wins_when_the_text_describes_another_slot(self) -> None:
        record = Mem0MemoryBackend._to_record(
            {
                "id": "x",
                "memory": "note: unrelated chatter",
                "metadata": {"key": "age", "value": "25"},
            }
        )
        assert (record.key, record.value) == ("age", "25")

    def test_category_and_importance_come_from_metadata(self) -> None:
        record = Mem0MemoryBackend._to_record(
            {
                "id": "x",
                "memory": "city: Dhaka",
                "metadata": {"key": "city", "value": "Dhaka", "category": "location",
                             "importance": 0.85},
            }
        )
        assert (record.category, record.importance) == (MemoryCategory.LOCATION, 0.85)


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


class TestLegacyMem0:
    @pytest.fixture
    def legacy(self, mem0_settings: Settings) -> Mem0MemoryBackend:
        return Mem0MemoryBackend(mem0_settings, memory=LegacyFakeMemory())

    def test_reads_fall_back_to_the_user_id_argument(
        self, legacy: Mem0MemoryBackend
    ) -> None:
        legacy.add(make_record("age", "25"))
        assert [r.value for r in legacy.all(TEST_USER)] == ["25"]

    def test_override_still_works_without_metadata_support(
        self, legacy: Mem0MemoryBackend
    ) -> None:
        """0.1.x cannot rewrite metadata, so the text has to be believed."""
        stored = legacy.add(make_record("age", "25"))
        legacy.update(stored.id, value="26")
        assert legacy.find_slot(TEST_USER, "identity", "age").value == "26"


# ---------------------------------------------------------------------------
# Through the handler
# ---------------------------------------------------------------------------


class TestThroughHandler:
    def test_a_changed_age_is_an_update_not_a_second_fact(
        self, mem0_settings: Settings, fake: FakeMemory
    ) -> None:
        llm = ScriptedLLM(
            json_queue=[
                facts_payload(("identity", "age", "25", 0.9)),
                facts_payload(("identity", "age", "26", 0.9)),
            ]
        )
        handler = MemoryHandler(
            Mem0MemoryBackend(mem0_settings, memory=fake), llm, settings=mem0_settings
        )
        handler.remember_turn(TEST_USER, "I am 25")
        writes = handler.remember_turn(TEST_USER, "Actually I am 26 now")

        assert [w.operation for w in writes] == [MemoryOperation.UPDATE]
        assert writes[0].previous_value == "25"
        assert [(r.key, r.value) for r in handler.profile(TEST_USER)] == [("age", "26")]
