"""Shared fixtures. No test touches the network or the user's real data dir."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from memory_chatbot.config import LLMProvider, MemoryBackendName, Settings
from memory_chatbot.llm.base import BaseLLMClient, PromptMessage
from memory_chatbot.memory.handler import MemoryHandler
from memory_chatbot.memory.sqlite_backend import SQLiteMemoryBackend
from memory_chatbot.prompts import MEMORY_CONTEXT_HEADER, NO_MEMORY_NOTE

TEST_USER = "test-user"


class ScriptedLLM(BaseLLMClient):
    """Deterministic stand-in for a provider.

    ``json_queue`` feeds :meth:`complete_json` (extraction / reconciliation) and
    ``text_queue`` feeds plain completions.

    Prompts are recorded in two streams because a single turn issues both kinds:
    the reply is generated first, then extraction runs. Tests about what the
    *chatbot* was shown must therefore look at ``chat_prompts``, not merely the
    last call made.
    """

    name = "scripted"

    def __init__(
        self,
        json_queue: Sequence[Any] = (),
        text_queue: Sequence[str] = (),
    ) -> None:
        super().__init__(model="scripted-1", temperature=0.0, max_tokens=256, timeout=5.0)
        self.json_queue = list(json_queue)
        self.text_queue = list(text_queue)
        #: Every call, in order, regardless of kind.
        self.prompts: list[list[PromptMessage]] = []
        #: Conversation (non-JSON) calls only.
        self.chat_prompts: list[list[PromptMessage]] = []
        #: Extraction / reconciliation (JSON) calls only.
        self.json_prompts: list[list[PromptMessage]] = []

    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        snapshot = [dict(m) for m in messages]
        self.prompts.append(snapshot)
        if overrides.get("response_format"):
            self.json_prompts.append(snapshot)
            payload = self.json_queue.pop(0) if self.json_queue else {"facts": []}
            return payload if isinstance(payload, str) else json.dumps(payload)
        self.chat_prompts.append(snapshot)
        return self.text_queue.pop(0) if self.text_queue else "ok"

    @property
    def last_prompt(self) -> list[PromptMessage]:
        """The most recent *conversation* prompt (what the chatbot was shown)."""
        return self.chat_prompts[-1] if self.chat_prompts else []

    def memory_block(self) -> str:
        """The injected memory-context system message from the last reply.

        Matched on the block header rather than the word "MEMORY", because the
        behaviour prompt also talks about the memory context.
        """
        return next(
            (
                m["content"]
                for m in self.last_prompt
                if m["role"] == "system"
                and m["content"].startswith((MEMORY_CONTEXT_HEADER, NO_MEMORY_NOTE))
            ),
            "",
        )


def facts_payload(*facts: tuple[str, str, str, float]) -> dict[str, Any]:
    """Build an extractor response: ``(category, key, value, importance)`` tuples."""
    return {
        "facts": [
            {"category": c, "key": k, "value": v, "importance": i} for c, k, v, i in facts
        ]
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at an isolated temp data dir, LLM extraction on."""
    return Settings(
        data_dir=tmp_path / "data",
        llm_provider=LLMProvider.ECHO,
        memory_backend=MemoryBackendName.SQLITE,
        memory_extraction_enabled=True,
        memory_top_k=6,
        memory_min_score=0.0,
        groq_api_key=None,
        google_api_key=None,
        openai_api_key=None,
    )


@pytest.fixture
def db_path(settings: Settings) -> Path:
    settings.ensure_directories()
    return settings.sqlite_path


@pytest.fixture
def backend(db_path: Path) -> Iterator[SQLiteMemoryBackend]:
    store = SQLiteMemoryBackend(db_path)
    yield store
    store.close()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def handler(backend: SQLiteMemoryBackend, llm: ScriptedLLM, settings: Settings) -> MemoryHandler:
    return MemoryHandler(backend, llm, settings=settings)
