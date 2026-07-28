"""``LLMConnector`` - the single door between the app and any LLM provider.

Responsibilities:
* build the prompt (system rules + recalled memory + recent history + question)
* call the configured provider, streaming when possible
* surface failures as a readable message instead of crashing the UI
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from memory_chatbot.config import DEFAULT_MODELS, LLMProvider, Settings, get_settings
from memory_chatbot.llm.base import BaseLLMClient, LLMError, PromptMessage, to_prompt_messages
from memory_chatbot.llm.providers import EchoClient, GeminiClient, GroqClient, OpenAIClient
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.models import ChatMessage, MemorySearchResult, Role
from memory_chatbot.prompts import SYSTEM_PROMPT, render_memory_context

logger = get_logger(__name__)

_PROVIDERS: dict[LLMProvider, type[BaseLLMClient]] = {
    LLMProvider.GROQ: GroqClient,
    LLMProvider.GEMINI: GeminiClient,
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.ECHO: EchoClient,
}


def build_llm_client(settings: Settings, provider: LLMProvider | None = None) -> BaseLLMClient:
    """Instantiate a provider client, falling back to :class:`EchoClient`.

    A missing key must not break the app: the user still gets a working UI plus
    a clear message about what to configure.
    """
    chosen = provider or settings.llm_provider
    if chosen is LLMProvider.ECHO:
        return EchoClient()

    # settings.llm_model only applies to the configured provider; a sidebar
    # switch to another one uses that provider's own default.
    model = settings.llm_model if chosen is settings.llm_provider else DEFAULT_MODELS[chosen]
    try:
        return _PROVIDERS[chosen](
            api_key=settings.api_key_for(chosen) or "",
            model=model or DEFAULT_MODELS[chosen],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    except LLMError as exc:
        logger.warning("Provider %s unavailable (%s); using offline echo mode", chosen, exc)
        return EchoClient()


class LLMConnector:
    """Sends queries to the configured LLM with memory context injected."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        *,
        settings: Settings | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or build_llm_client(self.settings)
        self.system_prompt = system_prompt

    # ------------------------------------------------------------- properties
    @property
    def provider_name(self) -> str:
        return self.client.name

    @property
    def model_name(self) -> str:
        return self.client.model

    @property
    def is_offline(self) -> bool:
        """True when no real provider is reachable (echo fallback in use)."""
        return isinstance(self.client, EchoClient)

    # ----------------------------------------------------------- prompt build
    def build_prompt(
        self,
        user_message: str,
        *,
        memories: Sequence[MemorySearchResult] = (),
        history: Sequence[ChatMessage] = (),
    ) -> list[PromptMessage]:
        """Assemble the full provider payload for one turn.

        Order matters: rules first, then long-term memory, then the short-term
        window, then the live question - so nothing outranks what was just asked.
        """
        messages: list[PromptMessage] = [
            {"role": str(Role.SYSTEM), "content": self.system_prompt},
            {"role": str(Role.SYSTEM), "content": render_memory_context(memories)},
        ]
        window = self.settings.history_window
        recent = [m for m in history if m.role is not Role.SYSTEM][-window:]
        messages.extend(to_prompt_messages(recent))
        messages.append({"role": str(Role.USER), "content": user_message})
        return messages

    # ------------------------------------------------------------- generation
    def generate(
        self,
        user_message: str,
        *,
        memories: Sequence[MemorySearchResult] = (),
        history: Sequence[ChatMessage] = (),
    ) -> str:
        """Return a complete reply (non-streaming)."""
        prompt = self.build_prompt(user_message, memories=memories, history=history)
        try:
            return self.client.complete(prompt).strip()
        except LLMError as exc:
            logger.error("Generation failed: %s", exc)
            return self._failure_message(exc)

    def stream(
        self,
        user_message: str,
        *,
        memories: Sequence[MemorySearchResult] = (),
        history: Sequence[ChatMessage] = (),
    ) -> Iterator[str]:
        """Yield reply chunks; errors are yielded as text so the UI stays alive."""
        prompt = self.build_prompt(user_message, memories=memories, history=history)
        try:
            yield from self.client.stream(prompt)
        except LLMError as exc:
            logger.error("Streaming failed: %s", exc)
            yield self._failure_message(exc)

    def _failure_message(self, exc: Exception) -> str:
        return (
            f"⚠️ I could not reach the {self.provider_name} API just now "
            f"({exc}). Your memories are safe on disk - retry, or switch "
            "provider in the sidebar."
        )
