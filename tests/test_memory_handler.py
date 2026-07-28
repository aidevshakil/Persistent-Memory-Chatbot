"""MemoryHandler: the add / update / delete / noop policy."""

from __future__ import annotations

import pytest

from memory_chatbot.memory.extraction import ConflictResolver
from memory_chatbot.memory.handler import MemoryHandler
from memory_chatbot.memory.sqlite_backend import SQLiteMemoryBackend
from memory_chatbot.models import CandidateFact, MemoryCategory, MemoryOperation

from .conftest import TEST_USER, ScriptedLLM, facts_payload


class TestRememberTurn:
    def test_a_new_fact_is_added(self, handler: MemoryHandler, llm: ScriptedLLM) -> None:
        llm.json_queue.append(facts_payload(("identity", "age", "25", 0.9)))
        (write,) = handler.remember_turn(TEST_USER, "I am 25")
        assert write.operation is MemoryOperation.ADD
        assert handler.count(TEST_USER) == 1

    def test_a_turn_with_no_facts_writes_nothing(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append({"facts": []})
        assert handler.remember_turn(TEST_USER, "What is the weather like?") == []
        assert handler.count(TEST_USER) == 0

    def test_repeating_a_known_fact_is_a_noop(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("identity", "age", "25", 0.9)))
        handler.remember_turn(TEST_USER, "I am 25")
        llm.json_queue.append(facts_payload(("identity", "age", "25", 0.9)))
        (write,) = handler.remember_turn(TEST_USER, "I'm 25 years old")
        assert write.operation is MemoryOperation.NOOP
        assert handler.count(TEST_USER) == 1

    def test_case_differences_still_count_as_known(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("location", "city", "Dhaka", 0.8)))
        handler.remember_turn(TEST_USER, "I live in Dhaka")
        llm.json_queue.append(facts_payload(("location", "city", "dhaka", 0.8)))
        (write,) = handler.remember_turn(TEST_USER, "I live in dhaka")
        assert write.operation is MemoryOperation.NOOP

    def test_multiple_facts_from_one_turn_are_all_stored(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        # The rule pass fires on this sentence too, using these same slot names,
        # so the merge must yield three facts rather than duplicating any.
        llm.json_queue.append(
            facts_payload(
                ("identity", "age", "25", 0.9),
                ("location", "city", "Dhaka", 0.8),
                ("interest", "interest", "football", 0.6),
            )
        )
        writes = handler.remember_turn(TEST_USER, "I'm 25, I live in Dhaka and I love football")
        assert {w.operation for w in writes} == {MemoryOperation.ADD}
        assert handler.count(TEST_USER) == 3


class TestOverrides:
    """The headline requirement: "I am 26 now" must replace 25, not sit beside it."""

    def test_new_age_replaces_the_old_one(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("identity", "age", "25", 0.9)))
        handler.remember_turn(TEST_USER, "I am 25")

        llm.json_queue.append(facts_payload(("identity", "age", "26", 0.9)))
        (write,) = handler.remember_turn(TEST_USER, "I am 26 now")

        assert write.operation is MemoryOperation.UPDATE
        assert write.previous_value == "25"
        assert handler.count(TEST_USER) == 1
        (record,) = handler.profile(TEST_USER)
        assert record.value == "26"
        assert record.history == ["25"]

    def test_a_move_replaces_the_stored_city(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("location", "city", "Dhaka", 0.8)))
        handler.remember_turn(TEST_USER, "I live in Dhaka")
        llm.json_queue.append(facts_payload(("location", "city", "Chittagong", 0.8)))
        (write,) = handler.remember_turn(TEST_USER, "I moved to Chittagong")
        assert write.operation is MemoryOperation.UPDATE
        assert handler.profile(TEST_USER)[0].value == "Chittagong"

    def test_a_synonymous_slot_updates_rather_than_duplicates(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        """"location" and "city" are the same single-valued attribute."""
        llm.json_queue.append(facts_payload(("location", "city", "Dhaka", 0.8)))
        handler.remember_turn(TEST_USER, "I live in Dhaka")
        llm.json_queue.append(facts_payload(("location", "location", "Chittagong", 0.8)))
        (write,) = handler.remember_turn(TEST_USER, "I am in Chittagong these days")
        assert write.operation is MemoryOperation.UPDATE
        assert handler.count(TEST_USER) == 1

    def test_independent_interests_coexist(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("interest", "sport", "football", 0.6)))
        handler.remember_turn(TEST_USER, "I play football")
        # Ambiguous case -> the resolver asks the model, which says "add".
        llm.json_queue.append(facts_payload(("interest", "hobby", "chess", 0.6)))
        llm.json_queue.append({"operation": "add", "target_id": None, "value": "chess"})
        (write,) = handler.remember_turn(TEST_USER, "I also play chess")
        assert write.operation is MemoryOperation.ADD
        assert handler.count(TEST_USER) == 2

    def test_an_importance_upgrade_is_kept(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        """An update never downgrades how load-bearing a fact is."""
        # A slot the regex rules do not cover, so the LLM's importance is used.
        llm.json_queue.append(facts_payload(("constraint", "budget", "tight", 0.4)))
        handler.remember_turn(TEST_USER, "Money is tight these days")
        llm.json_queue.append(facts_payload(("constraint", "budget", "very tight", 0.95)))
        handler.remember_turn(TEST_USER, "Money is extremely tight")
        assert handler.profile(TEST_USER)[0].importance == pytest.approx(0.95)


class TestRetraction:
    def test_retracting_a_fact_deletes_it(
        self, handler: MemoryHandler, llm: ScriptedLLM
    ) -> None:
        llm.json_queue.append(facts_payload(("interest", "favourite_sport", "football", 0.6)))
        handler.remember_turn(TEST_USER, "My favourite sport is football")
        assert handler.count(TEST_USER) == 1

        llm.json_queue.append(facts_payload(("interest", "favourite_sport", "football", 0.6)))
        (write,) = handler.remember_turn(TEST_USER, "I no longer play football")
        assert write.operation is MemoryOperation.DELETE
        assert handler.count(TEST_USER) == 0


class TestConflictResolver:
    def test_same_slot_same_value_is_a_noop(self, handler: MemoryHandler) -> None:
        handler.remember_fact(TEST_USER, category="identity", key="age", value="25")
        existing = handler.profile(TEST_USER)
        decision = ConflictResolver(ScriptedLLM(), use_llm=False).resolve(
            CandidateFact(MemoryCategory.IDENTITY, "age", "25"), existing
        )
        assert decision.operation is MemoryOperation.NOOP

    def test_an_empty_category_always_adds_without_calling_the_model(self) -> None:
        llm = ScriptedLLM()
        decision = ConflictResolver(llm).resolve(
            CandidateFact(MemoryCategory.GOAL, "goal", "learn guitar"), []
        )
        assert decision.operation is MemoryOperation.ADD
        assert llm.prompts == []

    def test_an_update_without_a_valid_target_degrades_to_add(
        self, handler: MemoryHandler
    ) -> None:
        """A model that says "update" but names no row cannot be trusted to clobber."""
        handler.remember_fact(TEST_USER, category="interest", key="sport", value="football")
        llm = ScriptedLLM([{"operation": "update", "target_id": "bogus-id", "value": "chess"}])
        decision = ConflictResolver(llm).resolve(
            CandidateFact(MemoryCategory.INTEREST, "hobby", "chess"),
            handler.profile(TEST_USER),
        )
        assert decision.operation is MemoryOperation.ADD

    def test_a_resolver_failure_keeps_the_fact(self, handler: MemoryHandler) -> None:
        handler.remember_fact(TEST_USER, category="interest", key="sport", value="football")
        llm = ScriptedLLM(["totally not json"])
        decision = ConflictResolver(llm).resolve(
            CandidateFact(MemoryCategory.INTEREST, "hobby", "chess"),
            handler.profile(TEST_USER),
        )
        assert decision.operation is MemoryOperation.ADD


class TestManualEntry:
    def test_remember_fact_normalises_its_input(self, handler: MemoryHandler) -> None:
        write = handler.remember_fact(
            TEST_USER, category="preference", key="Favourite Food", value="  I like Biryani  "
        )
        assert write.record.key == "favourite_food"
        assert write.record.value == "Biryani"

    def test_an_unknown_category_string_becomes_other(self, handler: MemoryHandler) -> None:
        write = handler.remember_fact(TEST_USER, category="nonsense", key="k", value="v")
        assert write.record.category is MemoryCategory.OTHER


class TestMaintenance:
    def test_profile_by_category_groups_records(self, handler: MemoryHandler) -> None:
        handler.remember_fact(TEST_USER, category="identity", key="age", value="25")
        handler.remember_fact(TEST_USER, category="identity", key="name", value="Shakil")
        handler.remember_fact(TEST_USER, category="location", key="city", value="Dhaka")
        grouped = handler.profile_by_category(TEST_USER)
        assert len(grouped[MemoryCategory.IDENTITY]) == 2
        assert len(grouped[MemoryCategory.LOCATION]) == 1

    def test_forget_all_wipes_one_profile_only(self, handler: MemoryHandler) -> None:
        handler.remember_fact(TEST_USER, category="identity", key="age", value="25")
        handler.remember_fact("other-user", category="identity", key="age", value="40")
        assert handler.forget_all(TEST_USER) == 1
        assert handler.count(TEST_USER) == 0
        assert handler.count("other-user") == 1

    def test_export_is_json_serialisable(self, handler: MemoryHandler) -> None:
        import json

        handler.remember_fact(TEST_USER, category="identity", key="age", value="25")
        payload = handler.export(TEST_USER)
        assert json.loads(json.dumps(payload))[0]["value"] == "25"

    def test_recall_never_raises_on_a_broken_backend(
        self, handler: MemoryHandler, backend: SQLiteMemoryBackend
    ) -> None:
        backend.close()  # simulate a dead connection
        assert handler.recall(TEST_USER, "anything") == []
