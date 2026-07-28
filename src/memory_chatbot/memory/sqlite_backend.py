"""Self-contained SQLite memory backend - the project's default store.

Two things make it fast enough to rank in Python:

* the embedding column is only read when an embedder is actually configured, and
* each user's tokenised corpus and IDF table are cached and reused across
  queries, invalidated on write.

Without those, every turn re-read and re-unpacked every stored vector.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import struct
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from memory_chatbot.llm.embeddings import BaseEmbedder, NullEmbedder, normalize
from memory_chatbot.logging_utils import get_logger
from memory_chatbot.memory.base import HISTORY_LIMIT, BaseMemoryBackend, MemoryBackendError
from memory_chatbot.memory.scoring import build_idf, document_tokens, score, tokenize
from memory_chatbot.models import MemoryCategory, MemoryRecord, MemorySearchResult, utc_now

logger = get_logger(__name__)

SCHEMA_VERSION = 1

#: How many users' corpora to keep prepared at once.
CORPUS_CACHE_SIZE = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    category       TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    importance     REAL NOT NULL DEFAULT 0.5,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    source_message TEXT,
    embedding      BLOB,
    history        TEXT NOT NULL DEFAULT '[]'
);

-- One live value per attribute: this is what makes "I am 26 now" an UPDATE
-- rather than a second row next to "25".
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_slot
    ON memories (user_id, category, key);
CREATE INDEX IF NOT EXISTS idx_memories_user_updated
    ON memories (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INSERT_COLUMNS = (
    "id, user_id, category, key, value, importance, created_at, updated_at, "
    "source_message, embedding, history"
)
#: Reads skip the BLOB unless embeddings are in use - it dominates row size.
_READ_COLUMNS = (
    "id, user_id, category, key, value, importance, created_at, updated_at, "
    "source_message, history"
)


def _pack(vector: Sequence[float] | None) -> bytes | None:
    return struct.pack(f"<{len(vector)}f", *vector) if vector else None


def _unpack(blob: bytes | None) -> list[float] | None:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob)) if blob else None


@dataclass(slots=True)
class _Corpus:
    """A user's memories, prepared for scoring.

    Tokenisation, the IDF table and vector normalisation all happen once here
    instead of on every query.
    """

    records: list[MemoryRecord]
    tokens: list[frozenset[str]]
    idf: dict[str, float]
    #: Unit-normalised embeddings, aligned with ``records``; all None when no
    #: embedder is configured.
    vectors: list[list[float] | None]


class SQLiteMemoryBackend(BaseMemoryBackend):
    """Durable memory store backed by a single SQLite file."""

    name = "sqlite"

    def __init__(self, db_path: str | Path, *, embedder: BaseEmbedder | None = None) -> None:
        self.db_path = Path(db_path)
        self.location = str(self.db_path)
        self.embedder = embedder or NullEmbedder()
        # Streamlit serves each interaction from a worker thread, so the
        # connection is shared explicitly and guarded by a lock.
        self._lock = threading.RLock()
        self._corpus: dict[str, _Corpus] = {}

        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                self.db_path, check_same_thread=False, isolation_level=None
            )
        except sqlite3.Error as exc:  # pragma: no cover - filesystem dependent
            raise MemoryBackendError(f"Cannot open memory database {self.db_path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ setup
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run one statement under the lock, translating driver errors.

        Callers above this layer only handle :class:`MemoryBackendError`, so a
        closed connection or a constraint violation must not surface as a raw
        ``sqlite3`` exception and break a chat turn.
        """
        with self._lock:
            try:
                return self._conn.execute(sql, params)
            except sqlite3.Error as exc:
                raise MemoryBackendError(f"Memory store operation failed: {exc}") from exc

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")  # non-blocking reads
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------ write
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Insert a record; an existing slot is updated instead of duplicated."""
        if existing := self.find_slot(record.user_id, str(record.category), record.key):
            return self.update(
                existing.id,
                value=record.value,
                importance=record.importance,
                embedding=record.embedding,
                source_message=record.source_message,
            )
        if record.embedding is None:
            record.embedding = self.embedder.embed_one(record.as_sentence())

        self._execute(
            f"INSERT INTO memories ({_INSERT_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.user_id,
                str(record.category),
                record.key,
                record.value,
                record.importance,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.source_message,
                _pack(record.embedding),
                json.dumps(record.history),
            ),
        )
        self._invalidate(record.user_id)
        logger.info("memory add | %s | %s", record.user_id, record.as_sentence())
        return record

    def update(
        self,
        record_id: str,
        *,
        value: str,
        importance: float | None = None,
        embedding: list[float] | None = None,
        source_message: str | None = None,
    ) -> MemoryRecord:
        """Overwrite a record's value, pushing the old one onto its history."""
        current = self.get(record_id)
        if current is None:
            raise MemoryBackendError(f"Cannot update unknown memory {record_id}")

        if value != current.value:
            current.history = [current.value, *current.history][:HISTORY_LIMIT]
        current.value = value
        current.importance = current.importance if importance is None else importance
        current.source_message = source_message or current.source_message
        current.updated_at = utc_now()
        current.embedding = embedding or self.embedder.embed_one(current.as_sentence())

        self._execute(
            "UPDATE memories SET value=?, importance=?, updated_at=?, "
            "source_message=?, embedding=?, history=? WHERE id=?",
            (
                current.value,
                current.importance,
                current.updated_at.isoformat(),
                current.source_message,
                _pack(current.embedding),
                json.dumps(current.history),
                record_id,
            ),
        )
        self._invalidate(current.user_id)
        logger.info("memory update | %s | %s", current.user_id, current.as_sentence())
        return current

    def delete(self, record_id: str) -> bool:
        deleted = self._execute("DELETE FROM memories WHERE id=?", (record_id,)).rowcount > 0
        if deleted:
            # The row's owner is already gone, so drop every prepared corpus.
            self._corpus.clear()
        return deleted

    def clear(self, user_id: str) -> int:
        removed = self._execute("DELETE FROM memories WHERE user_id=?", (user_id,)).rowcount
        self._invalidate(user_id)
        logger.info("memory clear | %s | %d removed", user_id, removed)
        return removed

    # ------------------------------------------------------------------- read
    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._execute(
            f"SELECT {_READ_COLUMNS}, embedding FROM memories WHERE id=?", (record_id,)
        ).fetchone()
        return self._to_record(row, with_embedding=True) if row else None

    def all(self, user_id: str) -> list[MemoryRecord]:
        return self._load(user_id)

    def find_slot(self, user_id: str, category: str, key: str) -> MemoryRecord | None:
        row = self._execute(
            f"SELECT {_READ_COLUMNS} FROM memories WHERE user_id=? AND category=? AND key=?",
            (user_id, str(category), key),
        ).fetchone()
        return self._to_record(row) if row else None

    def by_category(self, user_id: str, category: str) -> list[MemoryRecord]:
        """All live facts in one category - the conflict-resolution context."""
        rows = self._execute(
            f"SELECT {_READ_COLUMNS} FROM memories WHERE user_id=? AND category=? "
            "ORDER BY updated_at DESC",
            (user_id, str(category)),
        ).fetchall()
        return [self._to_record(row) for row in rows]

    def search(
        self, user_id: str, query: str, *, limit: int = 6, min_score: float = 0.0
    ) -> list[MemorySearchResult]:
        """Hybrid recall over one user's cached, pre-tokenised corpus."""
        corpus = self._corpus_for(user_id)
        if not corpus.records:
            return []

        query_tokens = tokenize(query)
        query_vector = (
            normalize(self.embedder.embed_one(query))
            if self.embedder.is_active and query.strip()
            else None
        )
        now = utc_now()

        results = [
            MemorySearchResult(
                record=record,
                score=score(
                    record,
                    doc_tokens=tokens,
                    query_tokens=query_tokens,
                    idf=corpus.idf,
                    now=now,
                    doc_vector=vector,
                    query_vector=query_vector,
                ),
            )
            for record, tokens, vector in zip(
                corpus.records, corpus.tokens, corpus.vectors, strict=True
            )
        ]
        results.sort(key=lambda item: (-item.score, -item.record.importance))
        return [item for item in results if item.score >= min_score][:limit]

    def users(self) -> list[str]:
        rows = self._execute("SELECT DISTINCT user_id FROM memories ORDER BY 1").fetchall()
        return [row["user_id"] for row in rows]

    def count(self, user_id: str) -> int:
        row = self._execute(
            "SELECT COUNT(*) AS n FROM memories WHERE user_id=?", (user_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        self._corpus.clear()
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()

    # ------------------------------------------------------------ corpus cache
    def _corpus_for(self, user_id: str) -> _Corpus:
        """Return the prepared corpus for *user_id*, building it if stale."""
        if cached := self._corpus.get(user_id):
            return cached

        semantic = self.embedder.is_active
        records = self._load(user_id, with_embeddings=semantic)
        tokens = [document_tokens(record) for record in records]
        corpus = _Corpus(
            records=records,
            tokens=tokens,
            idf=build_idf(tokens),
            vectors=[normalize(r.embedding) if semantic else None for r in records],
        )

        if len(self._corpus) >= CORPUS_CACHE_SIZE:
            self._corpus.pop(next(iter(self._corpus)))  # dicts are insertion-ordered
        self._corpus[user_id] = corpus
        return corpus

    def _invalidate(self, user_id: str) -> None:
        self._corpus.pop(user_id, None)

    # ---------------------------------------------------------------- mapping
    def _load(self, user_id: str, *, with_embeddings: bool = False) -> list[MemoryRecord]:
        columns = f"{_READ_COLUMNS}, embedding" if with_embeddings else _READ_COLUMNS
        rows = self._execute(
            f"SELECT {columns} FROM memories WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [self._to_record(row, with_embedding=with_embeddings) for row in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row, *, with_embedding: bool = False) -> MemoryRecord:
        return MemoryRecord.from_dict(
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "category": MemoryCategory.coerce(row["category"]),
                "key": row["key"],
                "value": row["value"],
                "importance": row["importance"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "source_message": row["source_message"],
                "embedding": _unpack(row["embedding"]) if with_embedding else None,
                "history": json.loads(row["history"] or "[]"),
            }
        )
