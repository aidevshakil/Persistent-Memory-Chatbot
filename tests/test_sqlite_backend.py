"""SQLite backend: CRUD, slot uniqueness, history and ranked recall."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from memory_chatbot.memory.base import MemoryBackendError
from memory_chatbot.memory.scoring import (
    build_idf,
    document_tokens,
    lexical_score,
    recency_score,
    tokenize,
)
from memory_chatbot.memory.sqlite_backend import CORPUS_CACHE_SIZE, SQLiteMemoryBackend
from memory_chatbot.models import MemoryCategory, MemoryRecord, utc_now

from .conftest import TEST_USER


def make_record(key: str, value: str, **kwargs) -> MemoryRecord:
    return MemoryRecord(
        user_id=kwargs.pop("user_id", TEST_USER),
        category=kwargs.pop("category", MemoryCategory.OTHER),
        key=key,
        value=value,
        **kwargs,
    )


class TestCrud:
    def test_add_then_get_round_trips(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("age", "25", category=MemoryCategory.IDENTITY))
        loaded = backend.get(stored.id)
        assert loaded is not None
        assert (loaded.key, loaded.value, loaded.category) == (
            "age", "25", MemoryCategory.IDENTITY,
        )

    def test_get_unknown_id_returns_none(self, backend: SQLiteMemoryBackend) -> None:
        assert backend.get("does-not-exist") is None

    def test_delete_removes_the_record(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        assert backend.delete(stored.id) is True
        assert backend.get(stored.id) is None
        assert backend.delete(stored.id) is False

    def test_update_of_unknown_id_raises(self, backend: SQLiteMemoryBackend) -> None:
        with pytest.raises(MemoryBackendError):
            backend.update("nope", value="x")

    def test_clear_only_touches_one_user(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("age", "25"))
        backend.add(make_record("age", "40", user_id="someone-else"))
        assert backend.clear(TEST_USER) == 1
        assert backend.count(TEST_USER) == 0
        assert backend.count("someone-else") == 1

    def test_users_lists_every_profile(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("age", "25", user_id="alice"))
        backend.add(make_record("age", "30", user_id="bob"))
        assert backend.users() == ["alice", "bob"]


class TestSlotUniqueness:
    def test_adding_the_same_slot_updates_instead_of_duplicating(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        first = backend.add(make_record("age", "25", category=MemoryCategory.IDENTITY))
        second = backend.add(make_record("age", "26", category=MemoryCategory.IDENTITY))
        assert second.id == first.id
        assert backend.count(TEST_USER) == 1
        assert backend.get(first.id).value == "26"

    def test_the_previous_value_is_archived(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="26")
        backend.update(stored.id, value="27")
        assert backend.get(stored.id).history == ["26", "25"]

    def test_history_is_capped(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("age", "1"))
        for value in range(2, 20):
            backend.update(stored.id, value=str(value))
        assert len(backend.get(stored.id).history) == 10

    def test_rewriting_the_same_value_does_not_grow_history(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        stored = backend.add(make_record("age", "25"))
        backend.update(stored.id, value="25")
        assert backend.get(stored.id).history == []

    def test_same_key_in_different_categories_coexists(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        backend.add(make_record("note", "a", category=MemoryCategory.GOAL))
        backend.add(make_record("note", "b", category=MemoryCategory.SKILL))
        assert backend.count(TEST_USER) == 2

    def test_find_slot_is_scoped_to_the_user(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("age", "25", category=MemoryCategory.IDENTITY))
        assert backend.find_slot(TEST_USER, "identity", "age") is not None
        assert backend.find_slot("other", "identity", "age") is None


class TestSearch:
    @pytest.fixture(autouse=True)
    def _seed(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("age", "25", category=MemoryCategory.IDENTITY, importance=0.9))
        backend.add(make_record("city", "Dhaka", category=MemoryCategory.LOCATION, importance=0.8))
        backend.add(
            make_record("favourite_sport", "football", category=MemoryCategory.INTEREST, importance=0.6)
        )
        backend.add(make_record("diet", "vegetarian", category=MemoryCategory.HEALTH, importance=0.7))

    def test_empty_store_returns_nothing(self, backend: SQLiteMemoryBackend) -> None:
        assert backend.search("nobody", "anything") == []

    def test_direct_term_match_ranks_first(self, backend: SQLiteMemoryBackend) -> None:
        results = backend.search(TEST_USER, "do I like football?")
        assert results[0].record.value == "football"

    def test_category_hint_lifts_the_right_fact(self, backend: SQLiteMemoryBackend) -> None:
        """"What sport should I try?" never says "football", yet must recall it."""
        results = backend.search(TEST_USER, "what sport should I try this weekend?")
        assert results[0].record.key == "favourite_sport"

    def test_diet_question_recalls_the_diet_fact(self, backend: SQLiteMemoryBackend) -> None:
        results = backend.search(TEST_USER, "suggest a recipe for dinner")
        assert "vegetarian" in [r.record.value for r in results[:2]]

    def test_limit_is_respected(self, backend: SQLiteMemoryBackend) -> None:
        assert len(backend.search(TEST_USER, "tell me about me", limit=2)) == 2

    def test_scores_are_normalised_and_descending(self, backend: SQLiteMemoryBackend) -> None:
        scores = [r.score for r in backend.search(TEST_USER, "where do I live?")]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_weak_matches(self, backend: SQLiteMemoryBackend) -> None:
        assert backend.search(TEST_USER, "quantum chromodynamics", min_score=0.99) == []


class TestScoringHelpers:
    def test_tokenize_drops_stopwords(self) -> None:
        assert tokenize("What sport should I try?") == ["sport", "try"]

    def test_rare_terms_get_more_weight_than_common_ones(self) -> None:
        docs = [
            document_tokens(make_record("s1", "football")),
            document_tokens(make_record("s2", "football")),
            document_tokens(make_record("s3", "football")),
            document_tokens(make_record("s4", "kabaddi")),
        ]
        idf = build_idf(docs)
        assert idf["kabaddi"] > idf["football"]

    def test_lexical_score_rewards_overlap(self) -> None:
        doc = document_tokens(make_record("favourite_sport", "football"))
        idf = build_idf([doc])
        assert lexical_score(tokenize("football"), doc, idf) > 0.5
        assert lexical_score(tokenize("astrophysics"), doc, idf) == 0.0

    def test_document_tokens_covers_key_and_value(self) -> None:
        tokens = document_tokens(make_record("favourite_sport", "football"))
        assert {"favourite", "sport", "football"} <= tokens

    def test_recency_decays_with_age(self) -> None:
        now = utc_now()
        assert recency_score(now, now=now) == pytest.approx(1.0, abs=1e-6)
        assert recency_score(now - timedelta(days=45), now=now) == pytest.approx(0.3679, abs=1e-3)
        assert recency_score(now - timedelta(days=400), now=now) < 0.01


class TestCorpusCache:
    """search() ranks over a cached corpus; every write must invalidate it."""

    def _values(self, backend: SQLiteMemoryBackend, query: str = "football") -> list[str]:
        return [r.record.value for r in backend.search(TEST_USER, query, limit=20)]

    def test_an_added_fact_is_visible_to_the_next_search(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        backend.add(make_record("sport", "football"))
        assert self._values(backend) == ["football"]
        backend.add(make_record("hobby", "chess"))
        assert set(self._values(backend)) == {"football", "chess"}

    def test_an_updated_value_replaces_the_cached_one(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        stored = backend.add(make_record("sport", "football"))
        self._values(backend)  # warm the cache
        backend.update(stored.id, value="cricket")
        assert self._values(backend) == ["cricket"]

    def test_a_deleted_fact_disappears_from_search(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("sport", "football"))
        self._values(backend)
        backend.delete(stored.id)
        assert self._values(backend) == []

    def test_clear_empties_the_cached_corpus(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("sport", "football"))
        self._values(backend)
        backend.clear(TEST_USER)
        assert self._values(backend) == []

    def test_caches_stay_separate_per_user(self, backend: SQLiteMemoryBackend) -> None:
        backend.add(make_record("sport", "football", user_id="alice"))
        backend.add(make_record("sport", "cricket", user_id="bob"))
        assert [r.record.value for r in backend.search("alice", "sport")] == ["football"]
        assert [r.record.value for r in backend.search("bob", "sport")] == ["cricket"]

    def test_eviction_does_not_lose_results(self, backend: SQLiteMemoryBackend) -> None:
        """More users than the cache holds must still rank correctly."""
        users = [f"user{i}" for i in range(CORPUS_CACHE_SIZE + 4)]
        for index, user in enumerate(users):
            backend.add(make_record("sport", f"sport-{index}", user_id=user))
        for index, user in enumerate(users):  # first users have been evicted
            assert [r.record.value for r in backend.search(user, "sport")] == [f"sport-{index}"]

    def test_repeated_searches_agree(self, backend: SQLiteMemoryBackend) -> None:
        """A cached corpus must rank identically, not just similarly."""
        backend.add(make_record("sport", "football", importance=0.6))
        backend.add(make_record("city", "Dhaka", category=MemoryCategory.LOCATION))
        first = backend.search(TEST_USER, "what sport should I try")
        second = backend.search(TEST_USER, "what sport should I try")

        assert [r.record.id for r in first] == [r.record.id for r in second]
        # Scores drift only by the recency term, which advances with the clock.
        for a, b in zip(first, second, strict=True):
            assert a.score == pytest.approx(b.score, abs=1e-6)


class TestDurability:
    def test_data_survives_reopening_the_database(self, db_path: Path) -> None:
        first = SQLiteMemoryBackend(db_path)
        stored = first.add(make_record("age", "25", category=MemoryCategory.IDENTITY))
        first.close()

        second = SQLiteMemoryBackend(db_path)
        try:
            reloaded = second.get(stored.id)
            assert reloaded is not None
            assert reloaded.value == "25"
        finally:
            second.close()

    def test_the_database_file_is_created_on_disk(self, db_path: Path) -> None:
        store = SQLiteMemoryBackend(db_path)
        try:
            store.add(make_record("age", "25"))
            assert db_path.exists() and db_path.stat().st_size > 0
        finally:
            store.close()

    def test_nested_directories_are_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "deeper" / "memory.db"
        store = SQLiteMemoryBackend(nested)
        try:
            assert nested.exists()
        finally:
            store.close()


class TestEmbeddingStorage:
    def test_vectors_round_trip_through_the_blob_column(
        self, backend: SQLiteMemoryBackend
    ) -> None:
        vector = [0.25, -0.5, 0.75]
        stored = backend.add(make_record("age", "25", embedding=vector))
        loaded = backend.get(stored.id)
        assert loaded.embedding == pytest.approx(vector)

    def test_absent_vectors_stay_none(self, backend: SQLiteMemoryBackend) -> None:
        stored = backend.add(make_record("age", "25"))
        assert backend.get(stored.id).embedding is None
