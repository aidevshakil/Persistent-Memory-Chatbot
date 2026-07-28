"""``ChatEngine`` - orchestrates recall -> generation -> memory write.

The sequence for one turn:

1. **Recall** the memories most relevant to the new message.
2. **Generate** a reply with those memories injected as system context.
3. **Remember** any durable facts the turn revealed.

Recall happens *before* generation (so the reply is informed) and the write
happens *after* (so the model never sees a fact it is about to be told, and a
failed generation does not pollute the store).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from memory_chatbot.config import Settings, get_settings
from memory_chatbot.llm.base import LLMError
from memory_chatbot.llm.connector import LLMConnector
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.memory.handler import MemoryHandler
from memory_chatbot.models import (
    ChatMessage,
    ChatTurnResult,
    MemorySearchResult,
    MemoryWrite,
    Role,
)
from memory_chatbot.prompts import PROFILE_SUMMARY_PROMPT

logger = get_logger(__name__)

DEFAULT_USER_ID = "default"


class ChatEngine:
    """Stateless-per-turn conversation orchestrator.

    Conversation history is passed in by the caller (the UI owns the session),
    while long-term memory lives in the handler - the split that lets a brand new
    session still know who the user is.
    """

    def __init__(
        self,
        connector: LLMConnector,
        memory: MemoryHandler,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.connector = connector
        self.memory = memory

    # -------------------------------------------------------------- properties
    @property
    def provider_name(self) -> str:
        return self.connector.provider_name

    @property
    def model_name(self) -> str:
        return self.connector.model_name

    @property
    def backend_name(self) -> str:
        return self.memory.backend_name

    @property
    def is_offline(self) -> bool:
        return self.connector.is_offline

    # -------------------------------------------------------------------- turn
    def recall(self, user_id: str, message: str) -> list[MemorySearchResult]:
        """Step 1: memories relevant to *message*."""
        return self.memory.recall(user_id, message)

    def respond(
        self,
        user_id: str,
        message: str,
        *,
        history: Sequence[ChatMessage] = (),
    ) -> ChatTurnResult:
        """Run a full turn and return the reply plus what memory did."""
        message = (message or "").strip()
        if not message:
            return ChatTurnResult(reply=ChatMessage(role=Role.ASSISTANT, content=""))

        recalled = self.recall(user_id, message)
        reply_text = self.connector.generate(message, memories=recalled, history=history)
        writes = self.remember(user_id, message, reply_text)
        return ChatTurnResult(
            reply=ChatMessage(
                role=Role.ASSISTANT,
                content=reply_text,
                used_memory_ids=[r.record.id for r in recalled],
            ),
            recalled=recalled,
            writes=writes,
        )

    def stream_response(
        self,
        user_id: str,
        message: str,
        *,
        history: Sequence[ChatMessage] = (),
        memories: Sequence[MemorySearchResult] | None = None,
    ) -> Iterator[str]:
        """Stream the reply text only.

        The UI calls :meth:`recall` first (to show which memories were used),
        passes them in here, then calls :meth:`remember` once the stream ends -
        keeping token-by-token rendering separate from the memory write.
        """
        recalled = list(memories) if memories is not None else self.recall(user_id, message)
        yield from self.connector.stream(message, memories=recalled, history=history)

    def remember(
        self, user_id: str, message: str, reply: str | None = None
    ) -> list[MemoryWrite]:
        """Step 3: persist durable facts from the turn."""
        try:
            writes = self.memory.remember_turn(user_id, message, reply)
        except Exception:
            logger.exception("Memory write failed for user %s", user_id)
            return []
        if persisted := [w for w in writes if w.operation.value != "noop"]:
            logger.info(
                "memory writes | %s | %s",
                user_id,
                "; ".join(w.describe() for w in persisted),
            )
        return writes

    # ----------------------------------------------------------------- profile
    def profile_summary(self, user_id: str) -> str:
        """One-paragraph "who is this user" summary for the sidebar."""
        records = self.memory.profile(user_id)
        if not records:
            return "No memories stored yet. Tell me something about yourself."
        facts = "\n".join(f"- {r.as_sentence()}" for r in records[:25])
        try:
            return self.connector.client.complete(
                [{"role": "user", "content": PROFILE_SUMMARY_PROMPT.format(facts=facts)}],
                temperature=0.2,
                max_tokens=200,
            ).strip()
        except LLMError as exc:
            logger.warning("Profile summary unavailable (%s); showing raw facts", exc)
            return facts

    def close(self) -> None:
        self.memory.close()


def build_chat_engine(settings: Settings | None = None) -> ChatEngine:
    """Assemble the whole application graph from configuration."""
    from memory_chatbot.memory.registry import build_memory_backend

    resolved = settings or get_settings()
    connector = LLMConnector(settings=resolved)
    backend = build_memory_backend(resolved)
    handler = MemoryHandler(backend, connector.client, settings=resolved)
    logger.info(
        "Chat engine ready | provider=%s model=%s memory=%s at %s",
        connector.provider_name,
        connector.model_name,
        handler.backend_name,
        handler.storage_location,
    )
    return ChatEngine(connector, handler, settings=resolved)
