"""Optional embedding providers used for semantic memory recall.

The app is designed to work with *no* embedding provider at all - the SQLite
backend then falls back to a lexical scorer. Configure ``EMBEDDING_PROVIDER``
to add vector similarity on top.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from memory_chatbot.config import EmbeddingProvider, Settings
from memory_chatbot.llm.base import LLMError
from memory_chatbot.logging_utils import get_logger

logger = get_logger(__name__)

Vector = list[float]


class BaseEmbedder(ABC):
    """Turns text into vectors."""

    name = "embedder"
    #: False for the null embedder, letting callers skip vector work entirely
    #: rather than reading and unpacking stored vectors nobody will compare.
    is_active = True

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of texts, preserving order."""

    def embed_one(self, text: str) -> Vector | None:
        if not text.strip():
            return None
        try:
            vectors = self.embed([text])
        except LLMError:
            logger.warning("Embedding failed; falling back to lexical recall", exc_info=True)
            return None
        return vectors[0] if vectors else None


class NullEmbedder(BaseEmbedder):
    """No-op embedder: keeps the app fully offline."""

    name = "none"
    is_active = False

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[] for _ in texts]

    def embed_one(self, text: str) -> Vector | None:
        return None


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("`openai` is required for EMBEDDING_PROVIDER=openai") from exc
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        try:
            response = self._client.embeddings.create(model=self.model, input=list(texts))
        except Exception as exc:
            raise LLMError(f"OpenAI embedding failed: {exc}") from exc
        return [item.embedding for item in response.data]


class GeminiEmbedder(BaseEmbedder):
    name = "gemini"

    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("`google-genai` is required for EMBEDDING_PROVIDER=gemini") from exc
        if not api_key:
            raise LLMError("GOOGLE_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        try:
            response = self._client.models.embed_content(model=self.model, contents=list(texts))
        except Exception as exc:
            raise LLMError(f"Gemini embedding failed: {exc}") from exc
        return [list(item.values or []) for item in (response.embeddings or [])]


def build_embedder(settings: Settings) -> BaseEmbedder:
    """Instantiate the configured embedder, degrading to :class:`NullEmbedder`."""
    provider = settings.embedding_provider
    if provider is EmbeddingProvider.NONE:
        return NullEmbedder()
    try:
        if provider is EmbeddingProvider.OPENAI:
            key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
            return OpenAIEmbedder(api_key=key, model=settings.embedding_model)
        key = settings.google_api_key.get_secret_value() if settings.google_api_key else ""
        return GeminiEmbedder(api_key=key, model=settings.embedding_model)
    except LLMError as exc:
        logger.warning("Embedder unavailable (%s); using lexical recall only", exc)
        return NullEmbedder()


# math.sumprod (3.12+) is a C-level fused dot product and roughly 5x faster than
# a generator expression over zip - the difference dominates semantic recall.
if hasattr(math, "sumprod"):  # pragma: no cover - version dependent
    _dot = math.sumprod
else:  # pragma: no cover - Python 3.11

    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))


def normalize(vector: Sequence[float] | None) -> list[float] | None:
    """Unit-length copy of *vector*; ``None`` if empty or zero-length.

    Normalising once at load time turns each later comparison into a single dot
    product instead of a dot product plus two norms.
    """
    if not vector:
        return None
    norm = math.sqrt(_dot(vector, vector))
    if norm == 0.0:
        return None
    return [x / norm for x in vector]


def unit_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Similarity of two *already normalised* vectors, mapped to ``[0, 1]``."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, (_dot(a, b) + 1.0) / 2.0))


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity mapped to ``[0, 1]``; ``0.0`` when either side is empty."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return unit_similarity(normalize(a), normalize(b))
