"""End-to-end persistence: the behaviour the project is actually judged on.

Each test simulates a *real restart* by tearing down every in-process object and
rebuilding from the same on-disk database - no shared state survives except the
files themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from memory_chatbot.chat.engine import ChatEngine
from memory_chatbot.chat.transcript import ConversationStore
from memory_chatbot.config import Settings
from memory_chatbot.llm.connector import LLMConnector
from memory_chatbot.memory.handler import MemoryHandler
from memory_chatbot.memory.sqlite_backend import SQLiteMemoryBackend
from memory_chatbot.models import ChatMessage, MemoryOperation, Role

from .conftest import TEST_USER, ScriptedLLM, facts_payload


class Session:
    """One "app run": its own connections, torn down on exit."""

    def __init__(self, settings: Settings, llm: ScriptedLLM) -> None:
        self.llm = llm
        self.backend = SQLiteMemoryBackend(settings.sqlite_path)
        self.handler = MemoryHandler(self.backend, llm, settings=settings)
        self.engine = ChatEngine(
            LLMConnector(llm, settings=settings), self.handler, settings=settings
        )
        self.transcript = ConversationStore(settings.sqlite_path)

    def close(self) -> None:
        self.engine.close()
        self.transcript.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def new_session(settings: Settings, json_queue: Sequence = (), text_queue: Sequence[str] = ()) -> Session:
    """Start a fresh "process" against the same database file."""
    return Session(settings, ScriptedLLM(json_queue, text_queue))


class TestFactsSurviveRestart:
    def test_a_fact_stated_yesterday_is_recalled_today(self, settings: Settings) -> None:
        """The assignment's core scenario, across a full restart."""
        # --- Day 1 -----------------------------------------------------------
        with new_session(
            settings,
            json_queue=[facts_payload(("interest", "favourite_sport", "football", 0.7))],
            text_queue=["Football is a great choice!"],
        ) as day1:
            day1.engine.respond(TEST_USER, "I like football")
            assert day1.handler.count(TEST_USER) == 1

        # --- Day 2: brand new objects, same database --------------------------
        with new_session(
            settings,
            json_queue=[{"facts": []}],
            text_queue=["Given how much you enjoy football, try a local five-a-side league."],
        ) as day2:
            result = day2.engine.respond(TEST_USER, "What sport should I try?")

            recalled = [item.record.value for item in result.recalled]
            assert "football" in recalled, "the stored interest was not recalled"
            # And it actually reached the model.
            assert "football" in day2.llm.memory_block()

    def test_several_facts_survive_two_restarts(self, settings: Settings) -> None:
        with new_session(
            settings,
            json_queue=[
                facts_payload(
                    ("identity", "age", "25", 0.9),
                    ("location", "city", "Dhaka", 0.8),
                    ("health", "diet", "vegetarian", 0.85),
                )
            ],
            text_queue=["Noted!"],
        ) as first:
            first.engine.respond(TEST_USER, "I'm 25, I live in Dhaka and I'm vegetarian")

        with new_session(settings) as second:
            assert second.handler.count(TEST_USER) == 3

        with new_session(settings) as third:
            stored = {r.key: r.value for r in third.handler.profile(TEST_USER)}
            assert stored == {"age": "25", "city": "Dhaka", "diet": "vegetarian"}

    def test_profiles_stay_isolated_across_restarts(self, settings: Settings) -> None:
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "25", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond("alice", "I am 25")
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "40", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond("bob", "I am 40")

        with new_session(settings) as run:
            assert [r.value for r in run.handler.profile("alice")] == ["25"]
            assert [r.value for r in run.handler.profile("bob")] == ["40"]


class TestOverrideSurvivesRestart:
    def test_a_new_age_replaces_the_old_one_across_sessions(self, settings: Settings) -> None:
        """"I am 26 now" must replace 25 - checked after a restart, not in-process."""
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "25", 0.9))], text_queue=["ok"]
        ) as day1:
            day1.engine.respond(TEST_USER, "I am 25 years old")

        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "26", 0.9))], text_queue=["ok"]
        ) as day2:
            result = day2.engine.respond(TEST_USER, "I am 26 now")
            assert [w.operation for w in result.writes] == [MemoryOperation.UPDATE]

        with new_session(settings) as day3:
            records = day3.handler.profile(TEST_USER)
            assert len(records) == 1, "the old age was kept alongside the new one"
            assert records[0].value == "26"
            assert records[0].history == ["25"], "the superseded value was not archived"

    def test_the_stale_value_is_never_shown_to_the_model_again(
        self, settings: Settings
    ) -> None:
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "25", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond(TEST_USER, "I am 25")
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "26", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond(TEST_USER, "I am 26 now")

        with new_session(settings, json_queue=[{"facts": []}], text_queue=["You are 26."]) as run:
            run.engine.respond(TEST_USER, "How old am I?")
            block = run.llm.memory_block()
            assert "26" in block
            assert "25" not in block

    def test_a_retraction_persists_as_a_deletion(self, settings: Settings) -> None:
        with new_session(
            settings,
            json_queue=[facts_payload(("interest", "favourite_sport", "football", 0.7))],
            text_queue=["ok"],
        ) as run:
            run.engine.respond(TEST_USER, "My favourite sport is football")

        with new_session(
            settings,
            json_queue=[facts_payload(("interest", "favourite_sport", "football", 0.7))],
            text_queue=["Understood."],
        ) as run:
            run.engine.respond(TEST_USER, "I no longer play football")

        with new_session(settings) as run:
            assert run.handler.count(TEST_USER) == 0


class TestTranscriptSurvivesRestart:
    def test_the_visible_conversation_reloads(self, settings: Settings) -> None:
        with new_session(settings) as run:
            run.transcript.append(TEST_USER, ChatMessage(role=Role.USER, content="I am 25"))
            run.transcript.append(TEST_USER, ChatMessage(role=Role.ASSISTANT, content="Noted!"))

        with new_session(settings) as run:
            history = run.transcript.load(TEST_USER)
            assert [(m.role, m.content) for m in history] == [
                (Role.USER, "I am 25"),
                (Role.ASSISTANT, "Noted!"),
            ]

    def test_transcripts_are_per_user(self, settings: Settings) -> None:
        with new_session(settings) as run:
            run.transcript.append("alice", ChatMessage(role=Role.USER, content="hi from alice"))
            run.transcript.append("bob", ChatMessage(role=Role.USER, content="hi from bob"))
            assert len(run.transcript.load("alice")) == 1
            assert run.transcript.load("bob")[0].content == "hi from bob"

    def test_clearing_the_chat_keeps_the_memories(self, settings: Settings) -> None:
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "25", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond(TEST_USER, "I am 25")
            run.transcript.append(TEST_USER, ChatMessage(role=Role.USER, content="I am 25"))
            run.transcript.clear(TEST_USER)
            assert run.transcript.load(TEST_USER) == []
            assert run.handler.count(TEST_USER) == 1


class TestPromptAssembly:
    def test_recalled_memories_are_injected_as_system_context(
        self, settings: Settings
    ) -> None:
        with new_session(
            settings, json_queue=[facts_payload(("location", "city", "Dhaka", 0.8))], text_queue=["ok"]
        ) as run:
            run.engine.respond(TEST_USER, "I live in Dhaka")

        with new_session(settings, json_queue=[{"facts": []}], text_queue=["Sunny."]) as run:
            run.engine.respond(TEST_USER, "What's the weather where I live?")
            prompt = run.llm.last_prompt
            assert prompt[0]["role"] == "system"          # behaviour rules
            assert "MEMORY CONTEXT" in prompt[1]["content"]  # recalled facts
            assert prompt[-1]["role"] == "user"           # the live question
            assert "Dhaka" in prompt[1]["content"]

    def test_an_empty_store_still_sends_a_memory_block(self, settings: Settings) -> None:
        with new_session(settings, json_queue=[{"facts": []}], text_queue=["Hello!"]) as run:
            run.engine.respond(TEST_USER, "Hi there")
            assert "empty" in run.llm.memory_block()

    def test_history_is_capped_to_the_configured_window(self, settings: Settings) -> None:
        narrow = settings.model_copy(update={"history_window": 2})
        with new_session(narrow, json_queue=[{"facts": []}], text_queue=["ok"]) as run:
            history = [
                ChatMessage(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content=f"msg {i}")
                for i in range(10)
            ]
            run.engine.respond(TEST_USER, "and now?", history=history)
            replayed = [m for m in run.llm.last_prompt if m["content"].startswith("msg ")]
            assert [m["content"] for m in replayed] == ["msg 8", "msg 9"]


class TestEngineResilience:
    def test_an_empty_message_is_a_no_op(self, settings: Settings) -> None:
        with new_session(settings) as run:
            result = run.engine.respond(TEST_USER, "   ")
            assert result.reply.content == ""
            assert result.writes == []

    def test_a_memory_write_failure_does_not_lose_the_reply(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with new_session(settings, json_queue=[{"facts": []}], text_queue=["Here you go."]) as run:
            monkeypatch.setattr(
                run.handler,
                "remember_turn",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")),
            )
            result = run.engine.respond(TEST_USER, "I am 25")
            assert result.reply.content == "Here you go."
            assert result.writes == []

    def test_the_database_file_lands_in_the_configured_data_dir(
        self, settings: Settings
    ) -> None:
        with new_session(
            settings, json_queue=[facts_payload(("identity", "age", "25", 0.9))], text_queue=["ok"]
        ) as run:
            run.engine.respond(TEST_USER, "I am 25")
        assert Path(settings.sqlite_path).is_file()
        assert Path(settings.sqlite_path).parent == Path(settings.data_dir)
