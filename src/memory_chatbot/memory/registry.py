"""Backend selection.

One place decides which store the app uses, so ``MEMORY_BACKEND=mem0`` is the
only change needed to switch persistence engines.
"""

from __future__ import annotations

from memory_chatbot.config import MemoryBackendName, Settings, get_settings
from memory_chatbot.llm.embeddings import BaseEmbedder, build_embedder
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.memory.base import BaseMemoryBackend, MemoryBackendError
from memory_chatbot.memory.sqlite_backend import SQLiteMemoryBackend

logger = get_logger(__name__)


def build_memory_backend(
    settings: Settings | None = None,
    *,
    embedder: BaseEmbedder | None = None,
) -> BaseMemoryBackend:
    """Instantiate the configured backend, falling back to SQLite.

    Mem0 needs an embedding provider and extra packages; if either is missing we
    log the reason and keep the app usable on the built-in store rather than
    failing to start.
    """
    resolved = settings or get_settings()
    resolved.ensure_directories()

    if resolved.memory_backend is MemoryBackendName.MEM0:
        try:
            from memory_chatbot.memory.mem0_backend import Mem0MemoryBackend

            return Mem0MemoryBackend(resolved)
        except MemoryBackendError as exc:
            logger.warning("Mem0 backend unavailable (%s); falling back to SQLite", exc)

    return SQLiteMemoryBackend(
        resolved.sqlite_path, embedder=embedder or build_embedder(resolved)
    )
