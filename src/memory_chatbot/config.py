"""Environment-driven application settings.

Every knob is read from the environment (or a local ``.env`` file) exactly once
and cached, so importing modules never re-parse the environment.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <root>/src/memory_chatbot/config.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env before Settings is instantiated so plain os.environ reads agree.
load_dotenv(find_dotenv(usecwd=True) or PROJECT_ROOT / ".env", override=False)


class LLMProvider(StrEnum):
    """Supported chat-completion providers."""

    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    ECHO = "echo"  # offline provider used by tests / no-API-key demos


class MemoryBackendName(StrEnum):
    """Supported memory persistence backends."""

    SQLITE = "sqlite"
    MEM0 = "mem0"


class EmbeddingProvider(StrEnum):
    """Embedding source used for semantic memory recall.

    ``none`` keeps the app fully offline: recall falls back to the lexical
    (BM25-style) scorer, which is enough for short first-person facts.
    """

    NONE = "none"
    OPENAI = "openai"
    GEMINI = "gemini"


#: Default chat model per provider - overridable with ``LLM_MODEL``.
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.GEMINI: "gemini-2.0-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ECHO: "echo-1",
}

#: Default embedding model per embedding provider.
DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.NONE: "",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.GEMINI: "text-embedding-004",
}


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Persistent Memory Chatbot"
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    log_level: str = "INFO"

    # ------------------------------------------------------------------ llm
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = ""
    llm_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # --------------------------------------------------------------- memory
    memory_backend: MemoryBackendName = MemoryBackendName.SQLITE
    memory_top_k: int = Field(default=6, gt=0, le=50)
    memory_min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Turn off to store raw messages only (useful when debugging extraction).
    memory_extraction_enabled: bool = True
    #: Recent turns replayed verbatim into the prompt (short-term window).
    history_window: int = Field(default=8, gt=0, le=50)

    embedding_provider: EmbeddingProvider = EmbeddingProvider.NONE
    embedding_model: str = ""

    # ------------------------------------------------------------ validators
    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _apply_model_defaults(self) -> Settings:
        if not self.llm_model:
            self.llm_model = DEFAULT_MODELS[self.llm_provider]
        if not self.embedding_model:
            self.embedding_model = DEFAULT_EMBEDDING_MODELS[self.embedding_provider]
        return self

    # ------------------------------------------------------------- derived
    @property
    def sqlite_path(self) -> Path:
        """On-disk location of the built-in memory store."""
        return self.data_dir / "memory.db"

    @property
    def mem0_dir(self) -> Path:
        """Directory Mem0 uses for its vector store + history db."""
        return self.data_dir / "mem0"

    def api_key_for(self, provider: LLMProvider) -> str | None:
        """Return the configured API key for *provider*, if any."""
        secret = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.GEMINI: self.google_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.ECHO: None,
        }[provider]
        return secret.get_secret_value() if secret else None

    def available_providers(self) -> list[LLMProvider]:
        """Providers that currently have credentials, plus the offline echo."""
        live = [p for p in LLMProvider if p is not LLMProvider.ECHO and self.api_key_for(p)]
        return [*live, LLMProvider.ECHO]

    def ensure_directories(self) -> None:
        """Create the persistent-storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.memory_backend is MemoryBackendName.MEM0:
            self.mem0_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests after patching the env)."""
    get_settings.cache_clear()
