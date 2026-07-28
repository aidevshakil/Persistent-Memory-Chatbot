"""Concrete LLM providers: Groq, Gemini, OpenAI and an offline echo client.

Vendor SDKs are imported lazily inside each constructor so the app starts even
when only one provider's package is installed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from memory_chatbot.llm.base import BaseLLMClient, LLMError, PromptMessage
from memory_chatbot.prompts import MEMORY_CONTEXT_HEADER, NO_MEMORY_NOTE

# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


class GroqClient(BaseLLMClient):
    """Groq chat completions (OpenAI-compatible schema, very low latency)."""

    name = "groq"

    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("Groq provider selected but `groq` is not installed") from exc
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = Groq(api_key=api_key, timeout=self.timeout)

    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        params = self._resolved(overrides)
        response_format = overrides.pop("response_format", None)
        try:
            completion = self._client.chat.completions.create(
                messages=list(messages),
                **params,
                **({"response_format": response_format} if response_format else {}),
            )
        except Exception as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc
        return completion.choices[0].message.content or ""

    def stream(self, messages: Sequence[PromptMessage], **overrides: Any) -> Iterator[str]:
        params = self._resolved(overrides)
        overrides.pop("response_format", None)
        try:
            stream = self._client.chat.completions.create(
                messages=list(messages), stream=True, **params
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                if delta := chunk.choices[0].delta.content:
                    yield delta
        except Exception as exc:
            raise LLMError(f"Groq stream failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GeminiClient(BaseLLMClient):
    """Google Gemini via the `google-genai` SDK."""

    name = "gemini"

    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("Gemini provider selected but `google-genai` is not installed") from exc
        if not api_key:
            raise LLMError("GOOGLE_API_KEY is not set")
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _split(self, messages: Sequence[PromptMessage]) -> tuple[str, list[Any]]:
        """Gemini takes system text separately and uses `model` for replies."""
        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            role, text = message["role"], message["content"]
            if role == "system":
                system_parts.append(text)
                continue
            contents.append(
                self._types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[self._types.Part.from_text(text=text)],
                )
            )
        return "\n\n".join(system_parts), contents

    def _config(self, params: dict[str, Any], system: str, *, as_json: bool) -> Any:
        return self._types.GenerateContentConfig(
            temperature=params["temperature"],
            max_output_tokens=params["max_tokens"],
            system_instruction=system or None,
            response_mime_type="application/json" if as_json else None,
        )

    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        as_json = bool(overrides.pop("response_format", None))
        params = self._resolved(overrides)
        system, contents = self._split(messages)
        try:
            response = self._client.models.generate_content(
                model=params["model"],
                contents=contents,
                config=self._config(params, system, as_json=as_json),
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return response.text or ""

    def stream(self, messages: Sequence[PromptMessage], **overrides: Any) -> Iterator[str]:
        overrides.pop("response_format", None)
        params = self._resolved(overrides)
        system, contents = self._split(messages)
        try:
            stream = self._client.models.generate_content_stream(
                model=params["model"],
                contents=contents,
                config=self._config(params, system, as_json=False),
            )
            for chunk in stream:
                if text := getattr(chunk, "text", None):
                    yield text
        except Exception as exc:
            raise LLMError(f"Gemini stream failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient(BaseLLMClient):
    """OpenAI chat completions."""

    name = "openai"

    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("OpenAI provider selected but `openai` is not installed") from exc
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, timeout=self.timeout)

    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        response_format = overrides.pop("response_format", None)
        params = self._resolved(overrides)
        try:
            completion = self._client.chat.completions.create(
                messages=list(messages),
                **params,
                **({"response_format": response_format} if response_format else {}),
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return completion.choices[0].message.content or ""

    def stream(self, messages: Sequence[PromptMessage], **overrides: Any) -> Iterator[str]:
        overrides.pop("response_format", None)
        params = self._resolved(overrides)
        try:
            stream = self._client.chat.completions.create(
                messages=list(messages), stream=True, **params
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                if delta := chunk.choices[0].delta.content:
                    yield delta
        except Exception as exc:
            raise LLMError(f"OpenAI stream failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Echo (offline)
# ---------------------------------------------------------------------------


class EchoClient(BaseLLMClient):
    """Key-free provider used for tests and for demoing the memory layer.

    It performs no generation: it restates the injected memory context so you
    can verify recall end-to-end without spending an API call.
    """

    name = "echo"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model", "echo-1")
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("max_tokens", 512)
        kwargs.setdefault("timeout", 5.0)
        super().__init__(**kwargs)

    def complete(self, messages: Sequence[PromptMessage], **overrides: Any) -> str:
        if overrides.get("response_format"):
            # The extractor must not invent facts in offline mode.
            return '{"facts": []}'
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        # Match the block header, not the word "MEMORY": the behaviour prompt
        # also talks about the memory context and would otherwise be picked up.
        memory_block = next(
            (
                m["content"]
                for m in messages
                if m["role"] == "system"
                and m["content"].startswith((MEMORY_CONTEXT_HEADER, NO_MEMORY_NOTE))
            ),
            "",
        )
        recalled = [
            line.removeprefix("-").strip()
            for line in memory_block.splitlines()
            if line.startswith("-")
        ]
        parts = [f'(offline echo mode) You said: "{question}"']
        if recalled:
            parts.append("Here is what I remember about you:")
            parts.extend(f"- {item}" for item in recalled)
        else:
            parts.append("I have no stored memories about you yet.")
        return "\n".join(parts)

    def stream(self, messages: Sequence[PromptMessage], **overrides: Any) -> Iterator[str]:
        for word in self.complete(messages, **overrides).split(" "):
            yield word + " "
