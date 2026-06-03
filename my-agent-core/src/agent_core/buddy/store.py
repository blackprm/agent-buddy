from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent_core.buddy.types import StoredCompanion
from agent_core.sqlite_utils import connect_sqlite

_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"


class BuddyStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = Path(os.getenv("AGENT_BUDDY_DB_PATH") or _DEFAULT_DB_DIR / "buddy.db")
        self._db_path = str(Path(db_path).expanduser())
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, row_factory=sqlite3.Row, pragmas=("PRAGMA journal_mode=WAL",))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS companions (
                    user_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    personality TEXT NOT NULL DEFAULT '',
                    hatched_at INTEGER NOT NULL,
                    muted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
            """)

    def get(self, *, user_id: str) -> StoredCompanion | None:
        with self._connect() as conn:
            row = conn.execute("SELECT name, personality, hatched_at, muted FROM companions WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return StoredCompanion(name=row["name"], personality=row["personality"], hatched_at=int(row["hatched_at"]), muted=bool(row["muted"]))

    def upsert(self, *, user_id: str, org_id: str, companion: StoredCompanion) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO companions (user_id, org_id, name, personality, hatched_at, muted, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    org_id = excluded.org_id,
                    name = excluded.name,
                    personality = excluded.personality,
                    hatched_at = excluded.hatched_at,
                    muted = excluded.muted,
                    updated_at = excluded.updated_at
            """, (user_id, org_id or "", companion.name, companion.personality, companion.hatched_at, 1 if companion.muted else 0, now))

    def set_muted(self, *, user_id: str, muted: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("UPDATE companions SET muted = ?, updated_at = ? WHERE user_id = ?", (1 if muted else 0, now, user_id))
