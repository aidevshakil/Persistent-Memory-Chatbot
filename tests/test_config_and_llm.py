"""Configuration defaults and the provider abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_chatbot.config import (
    DEFAULT_MODELS,
    EmbeddingProvider,
    LLMProvider,
    MemoryBackendName,
    Settings,
)
from memory_chatbot.llm.base import LLMError
from memory_chatbot.llm.connector import LLMConnector, build_llm_client
from memory_chatbot.llm.embeddings import NullEmbedder, unit_similarity
from memory_chatbot.llm.providers import EchoClient
from memory_chatbot.models import (
    ChatMessage,
    MemoryCategory,
    MemoryRecord,
    MemorySearchResult,
    Role,
)
from memory_chatbot.prompts import render_memory_context

from .conftest import ScriptedLLM


class TestSettings:
    def test_the_model_defaults_to_the_provider_default(self) -> None:
        assert Settings(llm_provider=LLMProvider.GROQ).llm_model == DEFAULT_MODELS[LLMProvider.GROQ]
        assert Settings(llm_provider=LLMProvider.GEMINI).llm_model == DEFAULT_MODELS[LLMProvider.GEMINI]

    def test_an_explicit_model_is_kept(self) -> None:
        assert Settings(llm_model="my-custom-model").llm_model == "my-custom-model"

    def test_derived_storage_paths(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path)
        assert settings.sqlite_path == tmp_path / "memory.db"
        assert settings.mem0_dir == tmp_path / "mem0"

    def test_ensure_directories_creates_the_data_dir(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path / "nested" / "data")
        settings.ensure_directories()
        assert settings.data_dir.is_dir()

    def test_echo_is_always_available_and_needs_no_key(self) -> None:
        settings = Settings(groq_api_key=None, google_api_key=None, openai_api_key=None)
        assert settings.available_providers() == [LLMProvider.ECHO]

    def test_providers_with_keys_are_listed_first(self) -> None:
        settings = Settings(groq_api_key="gsk_test", openai_api_key="sk-test")
        providers = settings.available_providers()
        assert providers[-1] is LLMProvider.ECHO
        assert set(providers[:-1]) == {LLMProvider.GROQ, LLMProvider.OPENAI}

    def test_api_key_lookup(self) -> None:
        settings = Settings(groq_api_key="gsk_secret")
        assert settings.api_key_for(LLMProvider.GROQ) == "gsk_secret"
        assert settings.api_key_for(LLMProvider.OPENAI) is None

    def test_keys_are_not_leaked_by_repr(self) -> None:
        assert "gsk_secret" not in repr(Settings(groq_api_key="gsk_secret"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("memory_top_k", 0), ("memory_min_score", 1.5), ("llm_temperature", 3.0), ("llm_max_tokens", 0)],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            Settings(**{field: value})

    def test_enum_fields_parse_from_strings(self) -> None:
        settings = Settings(
            llm_provider="openai", memory_backend="mem0", embedding_provider="gemini"
        )
        assert settings.llm_provider is LLMProvider.OPENAI
        assert settings.memory_backend is MemoryBackendName.MEM0
        assert settings.embedding_provider is EmbeddingProvider.GEMINI

    def test_log_level_is_upper_cased(self) -> None:
        assert Settings(log_level="debug").log_level == "DEBUG"


class TestProviderFallback:
    def test_a_missing_key_falls_back_to_offline_echo(self) -> None:
        client = build_llm_client(Settings(llm_provider=LLMProvider.GROQ, groq_api_key=None))
        assert isinstance(client, EchoClient)

    def test_the_connector_reports_offline_mode(self) -> None:
        connector = LLMConnector(settings=Settings(llm_provider=LLMProvider.OPENAI))
        assert connector.is_offline is True

    def test_echo_restates_the_recalled_memories(self) -> None:
        connector = LLMConnector(EchoClient(), settings=Settings())
        memories = [
            MemorySearchResult(
                MemoryRecord(
                    user_id="u",
                    category=MemoryCategory.INTEREST,
                    key="favourite_sport",
                    value="football",
                ),
                0.9,
            )
        ]
        reply = connector.generate("What sport should I try?", memories=memories)
        assert "favourite sport: football" in reply
        # Regression: the behaviour prompt also mentions "MEMORY CONTEXT", so a
        # naive substring match echoed the rules back instead of the facts.
        assert "Treat it as verified facts" not in reply

    def test_echo_says_so_when_nothing_is_stored(self) -> None:
        reply = LLMConnector(EchoClient(), settings=Settings()).generate("Who am I?")
        assert "no stored memories" in reply.lower()

    def test_echo_never_invents_facts_when_extracting(self) -> None:
        assert EchoClient().complete_json([{"role": "user", "content": "x"}]) == {"facts": []}


class TestConnector:
    def test_a_provider_error_becomes_a_readable_reply(self) -> None:
        class Broken(EchoClient):
            def complete(self, messages, **overrides):  # type: ignore[override]
                raise LLMError("rate limited")

        reply = LLMConnector(Broken(), settings=Settings()).generate("hi")
        assert "could not reach" in reply.lower()
        assert "memories are safe" in reply.lower()

    def test_a_streaming_error_is_yielded_as_text(self) -> None:
        class Broken(EchoClient):
            def stream(self, messages, **overrides):  # type: ignore[override]
                raise LLMError("connection reset")

        chunks = "".join(LLMConnector(Broken(), settings=Settings()).stream("hi"))
        assert "could not reach" in chunks.lower()

    def test_system_messages_from_history_are_dropped(self) -> None:
        llm = ScriptedLLM(text_queue=["ok"])
        history = [
            ChatMessage(role=Role.SYSTEM, content="internal note"),
            ChatMessage(role=Role.USER, content="earlier question"),
        ]
        LLMConnector(llm, settings=Settings()).generate("now what?", history=history)
        contents = [m["content"] for m in llm.last_prompt]
        assert "internal note" not in contents
        assert "earlier question" in contents


class TestMemoryContextRendering:
    def test_facts_are_grouped_by_category(self) -> None:
        results = [
            MemorySearchResult(
                MemoryRecord(user_id="u", category=MemoryCategory.IDENTITY, key="age", value="26"), 0.9
            ),
            MemorySearchResult(
                MemoryRecord(user_id="u", category=MemoryCategory.IDENTITY, key="name", value="Shakil"), 0.8
            ),
            MemorySearchResult(
                MemoryRecord(user_id="u", category=MemoryCategory.LOCATION, key="city", value="Dhaka"), 0.7
            ),
        ]
        block = render_memory_context(results)
        assert block.count("[identity]") == 1
        assert "[location]" in block
        assert "- age: 26" in block

    def test_no_memories_renders_an_explicit_empty_marker(self) -> None:
        assert "empty" in render_memory_context([])


class TestEmbeddingHelpers:
    def test_the_null_embedder_returns_nothing(self) -> None:
        assert NullEmbedder().embed_one("anything") is None

    def test_identical_vectors_score_one(self) -> None:
        assert unit_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_opposite_vectors_score_zero(self) -> None:
        assert unit_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("a", "b"), [(None, [1.0]), ([1.0], None), ([], []), ([1.0], [1.0, 2.0])]
    )
    def test_missing_or_mismatched_vectors_score_zero(self, a, b) -> None:
        assert unit_similarity(a, b) == 0.0


class TestLoggingSetup:
    def test_an_explicit_level_survives_later_lazy_imports(self) -> None:
        """Regression: a no-arg call from a lazily imported module reset the level.

        ``--quiet`` set ERROR, then importing a module inside a function called
        ``get_logger`` -> ``configure_logging()`` and put it back to INFO.
        """
        import logging

        from memory_chatbot.logging_utils import PACKAGE_LOGGER, configure_logging, get_logger

        logger = logging.getLogger(PACKAGE_LOGGER)
        previous = logger.level
        try:
            configure_logging("ERROR")
            get_logger("memory_chatbot.some.late.import")  # what a lazy import does
            assert logger.level == logging.ERROR
        finally:
            logger.setLevel(previous)

    def test_handlers_are_not_stacked_by_repeated_calls(self) -> None:
        """Streamlit re-runs the script on every interaction."""
        import logging

        from memory_chatbot.logging_utils import PACKAGE_LOGGER, configure_logging

        before = len(logging.getLogger(PACKAGE_LOGGER).handlers)
        for _ in range(5):
            configure_logging()
        assert len(logging.getLogger(PACKAGE_LOGGER).handlers) == before


class TestModels:
    def test_search_scores_are_clamped(self) -> None:
        record = MemoryRecord(user_id="u", category=MemoryCategory.OTHER, key="k", value="v")
        assert MemorySearchResult(record, 5.0).score == 1.0
        assert MemorySearchResult(record, -1.0).score == 0.0

    def test_records_round_trip_through_dicts(self) -> None:
        original = MemoryRecord(
            user_id="u",
            category=MemoryCategory.HEALTH,
            key="diet",
            value="vegan",
            importance=0.9,
            history=["vegetarian"],
        )
        restored = MemoryRecord.from_dict(original.to_dict())
        assert (restored.key, restored.value, restored.category, restored.history) == (
            "diet", "vegan", MemoryCategory.HEALTH, ["vegetarian"],
        )

    def test_timestamps_are_always_timezone_aware(self) -> None:
        restored = MemoryRecord.from_dict(
            {"user_id": "u", "category": "other", "key": "k", "value": "v", "created_at": "2024-01-01T10:00:00"}
        )
        assert restored.created_at.tzinfo is not None

    def test_an_unparseable_timestamp_does_not_crash(self) -> None:
        restored = MemoryRecord.from_dict(
            {"user_id": "u", "category": "other", "key": "k", "value": "v", "created_at": "not a date"}
        )
        assert restored.created_at.tzinfo is not None

    @pytest.mark.parametrize("value", ["identity", "IDENTITY", " Identity "])
    def test_category_coercion_is_forgiving(self, value: str) -> None:
        assert MemoryCategory.coerce(value) is MemoryCategory.IDENTITY

    @pytest.mark.parametrize("value", ["astrology", None, 42, ""])
    def test_unknown_categories_become_other(self, value: object) -> None:
        assert MemoryCategory.coerce(value) is MemoryCategory.OTHER

    def test_a_record_renders_as_a_readable_sentence(self) -> None:
        record = MemoryRecord(
            user_id="u", category=MemoryCategory.INTEREST, key="favourite_sport", value="football"
        )
        assert record.as_sentence() == "favourite sport: football"
