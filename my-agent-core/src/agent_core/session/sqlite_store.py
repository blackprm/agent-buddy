"""SQLite SessionStore 实现 — 零配置，文件级存储。

通过 register_session_store("sqlite", SQLiteSessionStore) 自动注册。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.session.store import SessionStore, serialize_message, deserialize_message
from agent_core.types import Message


_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"


class SQLiteSessionStore:
    """基于 SQLite 的会话持久化。

    零配置：默认在 ~/.my-agent-core/sessions.db 创建数据库。
    线程安全：每次操作获取新连接（SQLite WAL 模式）。

    参数:
        db_path: 数据库文件路径，默认 ~/.my-agent-core/sessions.db
        url: 兼容工厂接口，格式 sqlite:///path/to/db
    """

    def __init__(self, db_path: str | Path | None = None, *, url: str | None = None) -> None:
        if url:
            # 支持 sqlite:///path/to/db 格式
            if url.startswith("sqlite:///"):
                db_path = url[len("sqlite:///"):]
            else:
                db_path = url
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = _DEFAULT_DB_DIR / "sessions.db"
        self._db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL DEFAULT '',
                    org_id      TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata    TEXT NOT NULL DEFAULT '{}'
                )
            """)
            self._ensure_column(conn, "session", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "session", "org_id", "TEXT NOT NULL DEFAULT ''")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
                    turn        INTEGER NOT NULL DEFAULT 0,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_session_turn
                ON message(session_id, turn)
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_user ON session(user_id, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_org ON session(org_id, updated_at)")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def create_session(
        self,
        *,
        session_id: str | None = None,
        metadata: dict | None = None,
        user_id: str = "",
        org_id: str = "",
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session (id, user_id, org_id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (sid, user_id or "", org_id or "", now, now, meta_json),
            )
            if user_id or org_id:
                conn.execute(
                    "UPDATE session SET user_id = COALESCE(NULLIF(user_id, ''), ?), org_id = COALESCE(NULLIF(org_id, ''), ?) WHERE id = ?",
                    (user_id or "", org_id or "", sid),
                )
        return sid

    def save_message(self, session_id: str, message: Message, turn: int = 0) -> None:
        content_json = serialize_message(message)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO message (session_id, turn, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, turn, message.role, content_json, now),
            )
            conn.execute("UPDATE session SET updated_at = ? WHERE id = ?", (now, session_id))

    def save_messages(self, session_id: str, messages: list[Message], start_turn: int = 0) -> None:
        """批量保存：先删除 start_turn 及之后的消息，再重新插入。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM message WHERE session_id = ? AND turn >= ?",
                (session_id, start_turn),
            )
            for i, msg in enumerate(messages):
                turn = start_turn + i
                content_json = serialize_message(msg)
                conn.execute(
                    "INSERT INTO message (session_id, turn, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, turn, msg.role, content_json, now),
                )
            conn.execute("UPDATE session SET updated_at = ? WHERE id = ?", (now, session_id))

    def load_messages(self, session_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM message WHERE session_id = ? ORDER BY turn, id",
                (session_id,),
            ).fetchall()
        return [deserialize_message(row[0]) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, org_id, created_at, updated_at, metadata FROM session WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "user_id": row[1],
            "org_id": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "metadata": json.loads(row[5]),
        }

    def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if org_id is not None:
            clauses.append("org_id = ?")
            params.append(org_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, org_id, created_at, updated_at, metadata FROM session"
                f"{where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "org_id": r[2],
                "created_at": r[3],
                "updated_at": r[4],
                "metadata": json.loads(r[5]),
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM session WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def update_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "UPDATE session SET metadata = ?, updated_at = ? WHERE id = ?",
                (meta_json, now, session_id),
            )
