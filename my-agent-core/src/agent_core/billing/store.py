from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.sqlite_utils import connect_sqlite


_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"


@dataclass(slots=True)
class ModelPricing:
    model_id: str
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_read_per_million: float = 0.0
    cache_write_per_million: float = 0.0
    currency: str = "CNY"
    display_name: str = ""
    provider: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BillingStore:
    """SQLite-backed model pricing and per-session token/cost accounting."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = Path(os.getenv("AGENT_BILLING_DB_PATH") or _DEFAULT_DB_DIR / "billing.db")
        self._db_path = str(Path(db_path).expanduser())
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, row_factory=sqlite3.Row, pragmas=("PRAGMA journal_mode=WAL",))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_pricing (
                    model_id           TEXT PRIMARY KEY,
                    display_name       TEXT NOT NULL DEFAULT '',
                    provider           TEXT NOT NULL DEFAULT '',
                    input_per_million  REAL NOT NULL DEFAULT 0,
                    output_per_million REAL NOT NULL DEFAULT 0,
                    cache_read_per_million  REAL NOT NULL DEFAULT 0,
                    cache_write_per_million REAL NOT NULL DEFAULT 0,
                    currency           TEXT NOT NULL DEFAULT 'CNY',
                    updated_at         TEXT NOT NULL
                )
            """)
            self._ensure_column(conn, "model_pricing", "cache_read_per_million", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "model_pricing", "cache_write_per_million", "REAL NOT NULL DEFAULT 0")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_usage (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id          TEXT NOT NULL,
                    user_id             TEXT NOT NULL DEFAULT '',
                    org_id              TEXT NOT NULL DEFAULT '',
                    model_id            TEXT NOT NULL,
                    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
                    completion_tokens   INTEGER NOT NULL DEFAULT 0,
                    cache_read_input_tokens     INTEGER NOT NULL DEFAULT 0,
                    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens        INTEGER NOT NULL DEFAULT 0,
                    input_cost          REAL NOT NULL DEFAULT 0,
                    output_cost         REAL NOT NULL DEFAULT 0,
                    cache_read_cost     REAL NOT NULL DEFAULT 0,
                    cache_creation_cost REAL NOT NULL DEFAULT 0,
                    total_cost          REAL NOT NULL DEFAULT 0,
                    currency            TEXT NOT NULL DEFAULT 'CNY',
                    raw_usage           TEXT NOT NULL DEFAULT '{}',
                    created_at          TEXT NOT NULL
                )
            """)
            self._ensure_column(conn, "session_usage", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "session_usage", "org_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "session_usage", "cache_read_input_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "session_usage", "cache_creation_input_tokens", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "session_usage", "cache_read_cost", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "session_usage", "cache_creation_cost", "REAL NOT NULL DEFAULT 0")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_usage_session
                ON session_usage(session_id, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_usage_model
                ON session_usage(model_id, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_usage_user
                ON session_usage(user_id, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_usage_org
                ON session_usage(org_id, created_at)
            """)
            self._seed_default_models(conn)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _seed_default_models(self, conn: sqlite3.Connection) -> None:
        """Seed editable default pricing rows inspired by Claude Code's model catalog.

        Claude Code keeps model identity and price tiers as first-class model
        metadata (see its model configs / modelCost tables).  We mirror that
        shape here with a small built-in seed set, while still storing the
        source of truth in SQLite so Admin can edit or add local gateway models.
        """
        now = datetime.now(timezone.utc).isoformat()
        defaults = [
            # Claude public tiers: https://platform.claude.com/docs/about-claude/pricing
            # Format: id, name, provider, input, output, cache_read, cache_write, currency.
            # Claude cache read/write tiers follow Claude Code's modelCost table:
            # cache read = 10% input, cache write = 125% input.
            ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", "anthropic", 0.8, 4.0, 0.08, 1.0, "USD"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic", 1.0, 5.0, 0.1, 1.25, "USD"),
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", "anthropic", 3.0, 15.0, 0.3, 3.75, "USD"),
            ("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet", "anthropic", 3.0, 15.0, 0.3, 3.75, "USD"),
            ("claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", 3.0, 15.0, 0.3, 3.75, "USD"),
            ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "anthropic", 3.0, 15.0, 0.3, 3.75, "USD"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6", "anthropic", 3.0, 15.0, 0.3, 3.75, "USD"),
            ("claude-opus-4-20250514", "Claude Opus 4", "anthropic", 15.0, 75.0, 1.5, 18.75, "USD"),
            ("claude-opus-4-1-20250805", "Claude Opus 4.1", "anthropic", 15.0, 75.0, 1.5, 18.75, "USD"),
            ("claude-opus-4-5-20251101", "Claude Opus 4.5", "anthropic", 5.0, 25.0, 0.5, 6.25, "USD"),
            ("claude-opus-4-6", "Claude Opus 4.6", "anthropic", 5.0, 25.0, 0.5, 6.25, "USD"),
            ("fake", "Fake Debug Model", "fake", 0.0, 0.0, 0.0, 0.0, "CNY"),
            ("fake_tool", "Fake Tool Debug Model", "fake", 0.0, 0.0, 0.0, 0.0, "CNY"),
            ("fake_thinking", "Fake Thinking Debug Model", "fake", 0.0, 0.0, 0.0, 0.0, "CNY"),
        ]
        conn.executemany("""
            INSERT OR IGNORE INTO model_pricing
                (model_id, display_name, provider, input_per_million, output_per_million,
                 cache_read_per_million, cache_write_per_million, currency, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(*row, now) for row in defaults])
        conn.executemany("""
            UPDATE model_pricing
            SET cache_read_per_million = ?, cache_write_per_million = ?, updated_at = ?
            WHERE model_id = ?
              AND cache_read_per_million = 0
              AND cache_write_per_million = 0
        """, [(row[5], row[6], now, row[0]) for row in defaults if row[5] or row[6]])

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT model_id, display_name, provider, input_per_million,
                       output_per_million, cache_read_per_million,
                       cache_write_per_million, currency, updated_at
                FROM model_pricing
                ORDER BY provider, model_id
            """).fetchall()
        return [dict(row) for row in rows]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT model_id, display_name, provider, input_per_million,
                       output_per_million, cache_read_per_million,
                       cache_write_per_million, currency, updated_at
                FROM model_pricing WHERE model_id = ?
            """, (model_id,)).fetchone()
        return dict(row) if row else None

    def upsert_model(self, data: dict[str, Any]) -> dict[str, Any]:
        model_id = str(data.get("model_id") or data.get("id") or "").strip()
        if not model_id:
            raise ValueError("model_id is required")
        pricing = ModelPricing(
            model_id=model_id,
            display_name=str(data.get("display_name") or ""),
            provider=str(data.get("provider") or ""),
            input_per_million=float(data.get("input_per_million") or 0),
            output_per_million=float(data.get("output_per_million") or 0),
            cache_read_per_million=float(data.get("cache_read_per_million") or 0),
            cache_write_per_million=float(data.get("cache_write_per_million") or 0),
            currency=str(data.get("currency") or "CNY").upper(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO model_pricing
                    (model_id, display_name, provider, input_per_million, output_per_million,
                     cache_read_per_million, cache_write_per_million, currency, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    provider = excluded.provider,
                    input_per_million = excluded.input_per_million,
                    output_per_million = excluded.output_per_million,
                    cache_read_per_million = excluded.cache_read_per_million,
                    cache_write_per_million = excluded.cache_write_per_million,
                    currency = excluded.currency,
                    updated_at = excluded.updated_at
            """, (
                pricing.model_id,
                pricing.display_name,
                pricing.provider,
                pricing.input_per_million,
                pricing.output_per_million,
                pricing.cache_read_per_million,
                pricing.cache_write_per_million,
                pricing.currency,
                pricing.updated_at,
            ))
        return pricing.to_dict()

    def ensure_model(self, *, model_id: str, provider: str = "", display_name: str = "") -> dict[str, Any]:
        """Ensure a model row exists without overwriting existing pricing.

        Runtime-selected models can come from env/session config before the
        Admin pricing table knows about them.  This method registers such
        models with zero pricing so every model-related UI reads one table.
        """
        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("model_id is required")
        existing = self.get_model(model_id)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO model_pricing
                    (model_id, display_name, provider, input_per_million, output_per_million,
                     cache_read_per_million, cache_write_per_million, currency, updated_at)
                VALUES (?, ?, ?, 0, 0, 0, 0, 'CNY', ?)
            """, (
                model_id,
                display_name or model_id,
                provider,
                now,
            ))
        return self.get_model(model_id) or {
            "model_id": model_id,
            "display_name": display_name or model_id,
            "provider": provider,
            "input_per_million": 0.0,
            "output_per_million": 0.0,
            "cache_read_per_million": 0.0,
            "cache_write_per_million": 0.0,
            "currency": "CNY",
            "updated_at": now,
        }

    def delete_model(self, model_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM model_pricing WHERE model_id = ?", (model_id,))
            return cursor.rowcount > 0

    def delete_session_usage(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM session_usage WHERE session_id = ?", (session_id,))

    def record_usage(
        self,
        *,
        session_id: str,
        model_id: str,
        usage: dict[str, Any],
        user_id: str = "",
        org_id: str = "",
    ) -> dict[str, Any]:
        tokens = normalize_usage_token_breakdown(usage)
        prompt_tokens = tokens["prompt_tokens"]
        completion_tokens = tokens["completion_tokens"]
        cache_read_tokens = tokens["cache_read_input_tokens"]
        cache_creation_tokens = tokens["cache_creation_input_tokens"]
        total_tokens = tokens["total_tokens"]
        if total_tokens <= 0:
            return self.session_summary(session_id)
        pricing = self.get_model(model_id) or {
            "input_per_million": 0.0,
            "output_per_million": 0.0,
            "cache_read_per_million": 0.0,
            "cache_write_per_million": 0.0,
            "currency": "CNY",
        }
        input_cost = prompt_tokens * float(pricing.get("input_per_million") or 0) / 1_000_000
        output_cost = completion_tokens * float(pricing.get("output_per_million") or 0) / 1_000_000
        cache_read_cost = cache_read_tokens * float(pricing.get("cache_read_per_million") or 0) / 1_000_000
        cache_creation_cost = cache_creation_tokens * float(pricing.get("cache_write_per_million") or 0) / 1_000_000
        total_cost = input_cost + output_cost + cache_read_cost + cache_creation_cost
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO session_usage
                    (session_id, user_id, org_id, model_id, prompt_tokens, completion_tokens,
                     cache_read_input_tokens, cache_creation_input_tokens, total_tokens,
                     input_cost, output_cost, cache_read_cost, cache_creation_cost,
                     total_cost, currency, raw_usage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id or "",
                org_id or "",
                model_id,
                prompt_tokens,
                completion_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                total_tokens,
                input_cost,
                output_cost,
                cache_read_cost,
                cache_creation_cost,
                total_cost,
                str(pricing.get("currency") or "CNY"),
                json.dumps(usage, ensure_ascii=False),
                now,
            ))
        return self.session_summary(session_id)

    def usage_summary(self, *, user_id: str = "", org_id: str = "") -> dict[str, Any]:
        if user_id:
            where = "user_id = ?"
            value = user_id
        elif org_id:
            where = "org_id = ?"
            value = org_id
        else:
            return empty_billing_summary()
        with self._connect() as conn:
            row = conn.execute(f"""
                SELECT SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(cache_read_input_tokens) AS cache_read_input_tokens,
                       SUM(cache_creation_input_tokens) AS cache_creation_input_tokens,
                       SUM(total_tokens) AS total_tokens,
                       SUM(input_cost) AS input_cost,
                       SUM(output_cost) AS output_cost,
                       SUM(cache_read_cost) AS cache_read_cost,
                       SUM(cache_creation_cost) AS cache_creation_cost,
                       SUM(total_cost) AS total_cost,
                       COUNT(*) AS request_count,
                       GROUP_CONCAT(DISTINCT model_id) AS model_ids,
                       COALESCE(MAX(currency), 'CNY') AS currency
                FROM session_usage
                WHERE {where}
            """, (value,)).fetchone()
        return _summary_from_row(row)

    def session_summary(self, session_id: str) -> dict[str, Any]:
        summaries = self.session_summaries([session_id])
        return summaries.get(session_id, empty_billing_summary())

    def session_summaries(self, session_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT session_id,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(cache_read_input_tokens) AS cache_read_input_tokens,
                       SUM(cache_creation_input_tokens) AS cache_creation_input_tokens,
                       SUM(total_tokens) AS total_tokens,
                       SUM(input_cost) AS input_cost,
                       SUM(output_cost) AS output_cost,
                       SUM(cache_read_cost) AS cache_read_cost,
                       SUM(cache_creation_cost) AS cache_creation_cost,
                       SUM(total_cost) AS total_cost,
                       COUNT(*) AS request_count,
                       GROUP_CONCAT(DISTINCT model_id) AS model_ids,
                       COALESCE(MAX(currency), 'CNY') AS currency
                FROM session_usage
                WHERE session_id IN ({placeholders})
                GROUP BY session_id
            """, session_ids).fetchall()
        result = {sid: empty_billing_summary() for sid in session_ids}
        for row in rows:
            result[row["session_id"]] = {
                "prompt_tokens": int(row["prompt_tokens"] or 0),
                "completion_tokens": int(row["completion_tokens"] or 0),
                "cache_read_input_tokens": int(row["cache_read_input_tokens"] or 0),
                "cache_creation_input_tokens": int(row["cache_creation_input_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "input_cost": float(row["input_cost"] or 0),
                "output_cost": float(row["output_cost"] or 0),
                "cache_read_cost": float(row["cache_read_cost"] or 0),
                "cache_creation_cost": float(row["cache_creation_cost"] or 0),
                "total_cost": float(row["total_cost"] or 0),
                "currency": row["currency"] or "CNY",
                "request_count": int(row["request_count"] or 0),
                "model_ids": [m for m in str(row["model_ids"] or "").split(",") if m],
            }
        return result


def _summary_from_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None or int(row["request_count"] or 0) == 0:
        return empty_billing_summary()
    return {
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "cache_read_input_tokens": int(row["cache_read_input_tokens"] or 0),
        "cache_creation_input_tokens": int(row["cache_creation_input_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "input_cost": float(row["input_cost"] or 0),
        "output_cost": float(row["output_cost"] or 0),
        "cache_read_cost": float(row["cache_read_cost"] or 0),
        "cache_creation_cost": float(row["cache_creation_cost"] or 0),
        "total_cost": float(row["total_cost"] or 0),
        "currency": row["currency"] or "CNY",
        "request_count": int(row["request_count"] or 0),
        "model_ids": [m for m in str(row["model_ids"] or "").split(",") if m],
    }


def normalize_usage_tokens(usage: dict[str, Any]) -> tuple[int, int, int]:
    tokens = normalize_usage_token_breakdown(usage)
    return tokens["prompt_tokens"], tokens["completion_tokens"], tokens["total_tokens"]


def normalize_usage_token_breakdown(usage: dict[str, Any]) -> dict[str, int]:
    prompt = _int_token(usage, "prompt_tokens", "input_tokens")
    completion = _int_token(usage, "completion_tokens", "output_tokens")
    cache_read = _int_token(usage, "cache_read_input_tokens")
    cache_creation = _int_token(usage, "cache_creation_input_tokens")
    total = _int_token(usage, "total_tokens")
    if not total:
        total = prompt + completion + cache_read + cache_creation
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "total_tokens": total,
    }


def _int_token(data: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def empty_billing_summary() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cache_read_cost": 0.0,
        "cache_creation_cost": 0.0,
        "total_cost": 0.0,
        "currency": "CNY",
        "request_count": 0,
        "model_ids": [],
    }
