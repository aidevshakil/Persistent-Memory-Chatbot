"""Provider-agnostic LLM interface.

Every concrete provider implements :class:`BaseLLMClient`, so the rest of the
app never imports a vendor SDK. Swapping Groq for Gemini is a config change.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

from memory_chatbot.models import ChatMessage

#: A message as handed to a provider SDK.
PromptMessage = dict[str, str]


class LLMError(RuntimeError):
    """Raised when a provider call fails or is misconfigured."""


class BaseLLMClient(ABC):
    """Minimal contract the chatbot needs from a chat-completion provider."""

    #: Human-readable provider name, used in the UI and in error messages.
    name: str = "llm"

    def __init__(self, *, model: str, temperature: float, max_tokens: int, timeout: float) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------ interface
    @abstractmethod
    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        """Return the assistant reply for *messages*."""

    def stream(self, messages: Sequence[PromptMessage], **overrides: Any) -> Iterator[str]:
        """Yield reply chunks. Providers without streaming fall back to one chunk."""
        yield self.complete(messages, **overrides)

    # ------------------------------------------------------------- helpers
    def complete_json(
        self,
        messages: Sequence[PromptMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Any:
        """Call the model and parse its reply as JSON.

        Used by the memory extractor. Models sometimes wrap JSON in prose or a
        fenced block, so the response is repaired before parsing rather than
        failing the whole turn.
        """
        raw = self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens or self.max_tokens,
            response_format={"type": "json_object"},
        )
        return parse_json_response(raw)

    def _resolved(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge per-call overrides over the client defaults."""
        return {
            "model": overrides.pop("model", self.model),
            "temperature": overrides.pop("temperature", self.temperature),
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"


def to_prompt_messages(messages: Sequence[ChatMessage | PromptMessage]) -> list[PromptMessage]:
    """Normalise a mixed list of domain/raw messages into provider payloads."""
    out: list[PromptMessage] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            out.append(message.to_prompt_dict())
        else:
            out.append({"role": str(message["role"]), "content": str(message["content"])})
    return out


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_response(raw: str) -> Any:
    """Parse JSON from a model reply, tolerating fences and surrounding prose.

    Raises :class:`LLMError` if nothing JSON-shaped can be recovered.
    """
    text = (raw or "").strip()
    if not text:
        raise LLMError("Model returned an empty response where JSON was expected")

    candidates: list[str] = []
    if fenced := _FENCE_RE.search(text):
        candidates.append(fenced.group(1))
    candidates.append(text)
    # Last resort: the widest {...} / [...] span in the reply.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"Could not parse JSON from model response: {text[:200]!r}")
