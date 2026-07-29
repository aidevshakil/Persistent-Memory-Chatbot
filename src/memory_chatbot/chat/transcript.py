"""Durable conversation transcript.

Long-term *facts* are the point of the project, but reloading the visible
conversation on restart is what makes the persistence obvious to a user. This is
a separate, deliberately dumb store: raw turns, no extraction, no ranking.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path

from memory_chatbot.logging_utils import get_logger
from memory_chatbot.models import ChatMessage

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    used_memory_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_transcript_user ON transcript (user_id, created_at);
"""


class ConversationStore:
    """Append-only chat log, keyed by user."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def append(self, user_id: str, message: ChatMessage) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO transcript "
                "(id, user_id, role, content, created_at, used_memory_ids) "
                "VALUES (?,?,?,?,?,?)",
                (
                    message.id,
                    user_id,
                    str(message.role),
                    message.content,
                    message.created_at.isoformat(),
                    json.dumps(message.used_memory_ids),
                ),
            )

    def load(self, user_id: str, *, limit: int = 200) -> list[ChatMessage]:
        """Most recent *limit* messages for *user_id*, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, created_at, used_memory_ids FROM transcript "
                "WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        messages = [
            ChatMessage.from_dict(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "used_memory_ids": json.loads(row["used_memory_ids"] or "[]"),
                }
            )
            for row in rows
        ]
        messages.reverse()  # query was newest-first; the UI renders oldest-first
        return messages

    def clear(self, user_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM transcript WHERE user_id=?", (user_id,))
        return cursor.rowcount

    def users(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM transcript ORDER BY 1"
            ).fetchall()
        return [row["user_id"] for row in rows]

    def close(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()
