"""Fact extraction: rules, LLM parsing, and the merge between them."""

from __future__ import annotations

import pytest

from memory_chatbot.llm.base import LLMError, parse_json_response
from memory_chatbot.memory.extraction import (
    FactExtractor,
    is_single_valued,
    looks_like_retraction,
    normalise_key,
    normalise_value,
    rule_based_facts,
)
from memory_chatbot.models import MemoryCategory

from .conftest import ScriptedLLM, facts_payload


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Favourite Sport", "favourite_sport"),
            ("  age  ", "age"),
            ("job-title!", "job_title"),
            ("", "note"),
            ("___", "note"),
        ],
    )
    def test_keys_are_normalised(self, raw: str, expected: str) -> None:
        assert normalise_key(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("I am 26 years old", "26 years old"),
            ("  football  ", "football"),
            ("I like  spicy   food.", "spicy food"),
            ("My city is Dhaka", "city is Dhaka"),
        ],
    )
    def test_values_drop_first_person_phrasing(self, raw: str, expected: str) -> None:
        assert normalise_value(raw) == expected

    def test_values_are_length_capped(self) -> None:
        assert len(normalise_value("x" * 5000)) == 300


class TestSingleValuedSlots:
    @pytest.mark.parametrize("key", ["age", "name", "city", "job_title", "diet"])
    def test_known_attributes_are_single_valued(self, key: str) -> None:
        assert is_single_valued(key)

    @pytest.mark.parametrize("key", ["favourite_sport", "primary_language", "current_goal"])
    def test_superlative_prefixes_are_single_valued(self, key: str) -> None:
        assert is_single_valued(key)

    @pytest.mark.parametrize("key", ["interest", "skill", "hobby", "pet"])
    def test_multi_valued_attributes_are_not(self, key: str) -> None:
        assert not is_single_valued(key)


class TestRuleBasedExtraction:
    def test_extracts_age_location_and_interest(self) -> None:
        facts = {f.key: f for f in rule_based_facts("I'm 25 and I live in Dhaka. I love football")}
        assert facts["age"].value == "25"
        assert facts["age"].category is MemoryCategory.IDENTITY
        assert facts["city"].value == "Dhaka"
        assert facts["interest"].value == "football"

    def test_extracts_age_from_i_turned(self) -> None:
        facts = {f.key: f.value for f in rule_based_facts("I turned 26 last week")}
        assert facts["age"] == "26"

    def test_extracts_allergy_with_high_importance(self) -> None:
        (fact,) = [f for f in rule_based_facts("I am allergic to peanuts") if f.key == "allergy"]
        assert fact.value == "peanuts"
        assert fact.importance >= 0.9

    def test_trailing_conjunctions_are_trimmed(self) -> None:
        facts = {f.key: f.value for f in rule_based_facts("I live in Dhaka and work remotely")}
        assert facts["city"] == "Dhaka"

    def test_a_following_clause_after_a_comma_is_trimmed(self) -> None:
        """Regression: the greedy city pattern swallowed the next clause."""
        facts = {
            f.key: f.value
            for f in rule_based_facts("I am 25, I live in Dhaka, I love football")
        }
        assert facts["city"] == "Dhaka"
        assert facts["age"] == "25"
        assert facts["interest"] == "football"

    def test_a_legitimate_comma_inside_a_value_is_kept(self) -> None:
        """"Dhaka, Bangladesh" is one value - only a new subject splits it."""
        facts = {f.key: f.value for f in rule_based_facts("I live in Dhaka, Bangladesh")}
        assert facts["city"] == "Dhaka, Bangladesh"

    def test_a_sentence_boundary_stops_the_match(self) -> None:
        facts = {f.key: f.value for f in rule_based_facts("I live in Dhaka. I love football")}
        assert facts["city"] == "Dhaka"

    @pytest.mark.parametrize(
        "message",
        [
            "What sport should I try?",
            "Tell me a joke",
            "How does memory work?",
            "hi",
        ],
    )
    def test_questions_and_small_talk_store_nothing(self, message: str) -> None:
        assert rule_based_facts(message) == []


class TestRetractionDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "I no longer play football",
            "I don't live in Dhaka",
            "forget what I said about football",
            "I quit my job",
        ],
    )
    def test_retractions_are_detected(self, message: str) -> None:
        assert looks_like_retraction(message)

    @pytest.mark.parametrize("message", ["I love football", "I am 26 now", "What's up?"])
    def test_normal_statements_are_not_retractions(self, message: str) -> None:
        assert not looks_like_retraction(message)


class TestJSONRepair:
    def test_parses_plain_json(self) -> None:
        assert parse_json_response('{"facts": []}') == {"facts": []}

    def test_parses_fenced_json(self) -> None:
        raw = 'Sure!\n```json\n{"facts": [{"key": "age"}]}\n```\nHope that helps.'
        assert parse_json_response(raw)["facts"][0]["key"] == "age"

    def test_parses_json_surrounded_by_prose(self) -> None:
        assert parse_json_response('Here you go: {"a": 1} - done') == {"a": 1}

    @pytest.mark.parametrize("raw", ["", "   ", "no json at all"])
    def test_unparseable_replies_raise(self, raw: str) -> None:
        with pytest.raises(LLMError):
            parse_json_response(raw)


class TestFactExtractor:
    def test_uses_llm_output(self) -> None:
        llm = ScriptedLLM([facts_payload(("goal", "fitness_goal", "run a marathon", 0.7))])
        (fact,) = FactExtractor(llm).extract("I want to run a marathon next year")
        assert fact.key == "fitness_goal"
        assert fact.category is MemoryCategory.GOAL

    def test_rules_win_over_llm_on_the_same_slot(self) -> None:
        """A regex cannot hallucinate an age; a model can round one."""
        llm = ScriptedLLM([facts_payload(("identity", "age", "twenty five", 0.9))])
        facts = {f.key: f.value for f in FactExtractor(llm).extract("I am 25")}
        assert facts["age"] == "25"

    def test_llm_failure_falls_back_to_rules(self) -> None:
        llm = ScriptedLLM(["not json at all"])
        facts = {f.key: f.value for f in FactExtractor(llm).extract("I am 30 years old")}
        assert facts["age"] == "30"

    def test_unknown_category_becomes_other(self) -> None:
        llm = ScriptedLLM([facts_payload(("astrology", "star_sign", "leo", 0.4))])
        (fact,) = FactExtractor(llm).extract("My star sign is Leo")
        assert fact.category is MemoryCategory.OTHER

    def test_malformed_entries_are_skipped(self) -> None:
        llm = ScriptedLLM(
            [{"facts": ["not a dict", {"key": "", "value": ""}, {"key": "city", "value": "Dhaka"}]}]
        )
        facts = FactExtractor(llm).extract("Something about Dhaka")
        assert [f.value for f in facts] == ["Dhaka"]

    def test_non_numeric_importance_defaults(self) -> None:
        llm = ScriptedLLM([{"facts": [{"key": "hobby", "value": "chess", "importance": "high"}]}])
        (fact,) = FactExtractor(llm).extract("I play chess")
        assert fact.importance == 0.5

    def test_llm_disabled_uses_rules_only(self) -> None:
        llm = ScriptedLLM([facts_payload(("goal", "goal", "should not appear", 0.9))])
        facts = {f.key: f.value for f in FactExtractor(llm, use_llm=False).extract("I am 41")}
        assert facts == {"age": "41"}
        assert llm.prompts == []

    @pytest.mark.parametrize("message", ["", "  ", "hi"])
    def test_trivial_messages_are_ignored(self, message: str) -> None:
        assert FactExtractor(ScriptedLLM()).extract(message) == []
