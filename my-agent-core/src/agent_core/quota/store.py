from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent_core.sqlite_utils import connect_sqlite


_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"


@dataclass(slots=True)
class QuotaDecision:
    allowed: bool
    reason: str = ""
    user_id: str = ""
    org_id: str = ""
    status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuotaStore:
    """Server-side quota policy and counter store.

    Policies can exist at default, user, or org scope.  Limits are zero when
    unlimited.  Preflight checks block only when an already-known counter has
    reached a limit; post-call usage recording advances the server-side truth.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = Path(os.getenv("AGENT_QUOTA_DB_PATH") or _DEFAULT_DB_DIR / "quota.db")
        self._db_path = str(Path(db_path).expanduser())
        self._init_db()

    @property
    def enabled(self) -> bool:
        return os.getenv("AGENT_QUOTA_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, row_factory=sqlite3.Row, pragmas=("PRAGMA journal_mode=WAL",))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quota_policies (
                    id                     TEXT PRIMARY KEY,
                    scope_type             TEXT NOT NULL,
                    scope_id               TEXT NOT NULL DEFAULT '',
                    name                   TEXT NOT NULL DEFAULT '',
                    max_requests_per_day   INTEGER NOT NULL DEFAULT 0,
                    max_tokens_per_day     INTEGER NOT NULL DEFAULT 0,
                    max_cost_per_day       REAL NOT NULL DEFAULT 0,
                    max_requests_per_week  INTEGER NOT NULL DEFAULT 0,
                    max_tokens_per_week    INTEGER NOT NULL DEFAULT 0,
                    max_cost_per_week      REAL NOT NULL DEFAULT 0,
                    max_requests_per_month INTEGER NOT NULL DEFAULT 0,
                    max_tokens_per_month   INTEGER NOT NULL DEFAULT 0,
                    max_cost_per_month     REAL NOT NULL DEFAULT 0,
                    enabled                INTEGER NOT NULL DEFAULT 1,
                    metadata               TEXT NOT NULL DEFAULT '{}',
                    created_at             TEXT NOT NULL,
                    updated_at             TEXT NOT NULL,
                    UNIQUE(scope_type, scope_id)
                )
            """)
            self._ensure_column(conn, "quota_policies", "max_requests_per_week", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "quota_policies", "max_tokens_per_week", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "quota_policies", "max_cost_per_week", "REAL NOT NULL DEFAULT 0")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quota_counters (
                    scope_type    TEXT NOT NULL,
                    scope_id      TEXT NOT NULL DEFAULT '',
                    window        TEXT NOT NULL,
                    window_start  TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    total_tokens  INTEGER NOT NULL DEFAULT 0,
                    total_cost    REAL NOT NULL DEFAULT 0,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY(scope_type, scope_id, window, window_start)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quota_counters_scope ON quota_counters(scope_type, scope_id)")
            self.ensure_default_policy(conn=conn)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def ensure_default_policy(self, *, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
        close = conn is None
        conn = conn or self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            policy = {
                "scope_type": "default",
                "scope_id": "",
                "name": "Default local quota",
                "max_requests_per_day": _env_int("AGENT_DEFAULT_DAILY_REQUEST_LIMIT"),
                "max_tokens_per_day": _env_int("AGENT_DEFAULT_DAILY_TOKEN_LIMIT"),
                "max_cost_per_day": _env_float("AGENT_DEFAULT_DAILY_COST_LIMIT"),
                "max_requests_per_week": _env_int("AGENT_DEFAULT_WEEKLY_REQUEST_LIMIT"),
                "max_tokens_per_week": _env_int("AGENT_DEFAULT_WEEKLY_TOKEN_LIMIT"),
                "max_cost_per_week": _env_float("AGENT_DEFAULT_WEEKLY_COST_LIMIT"),
                "max_requests_per_month": _env_int("AGENT_DEFAULT_MONTHLY_REQUEST_LIMIT"),
                "max_tokens_per_month": _env_int("AGENT_DEFAULT_MONTHLY_TOKEN_LIMIT"),
                "max_cost_per_month": _env_float("AGENT_DEFAULT_MONTHLY_COST_LIMIT"),
                "metadata": {"source": "env", "zero_means_unlimited": True},
            }
            conn.execute("""
                INSERT OR IGNORE INTO quota_policies
                    (id, scope_type, scope_id, name, max_requests_per_day, max_tokens_per_day,
                     max_cost_per_day, max_requests_per_week, max_tokens_per_week, max_cost_per_week,
                     max_requests_per_month, max_tokens_per_month,
                     max_cost_per_month, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "default",
                policy["scope_type"],
                policy["scope_id"],
                policy["name"],
                policy["max_requests_per_day"],
                policy["max_tokens_per_day"],
                policy["max_cost_per_day"],
                policy["max_requests_per_week"],
                policy["max_tokens_per_week"],
                policy["max_cost_per_week"],
                policy["max_requests_per_month"],
                policy["max_tokens_per_month"],
                policy["max_cost_per_month"],
                json.dumps(policy["metadata"], ensure_ascii=False),
                now,
                now,
            ))
            return self.get_policy(scope_type="default", scope_id="", conn=conn) or policy
        finally:
            if close:
                conn.close()

    def upsert_policy(self, data: dict[str, Any]) -> dict[str, Any]:
        scope_type = str(data.get("scope_type") or data.get("scopeType") or "default").strip()
        scope_id = str(data.get("scope_id") or data.get("scopeId") or "").strip()
        if scope_type not in {"default", "user", "org"}:
            raise ValueError("scope_type must be default, user, or org")
        if scope_type != "default" and not scope_id:
            raise ValueError("scope_id is required for user/org quota policies")
        now = datetime.now(timezone.utc).isoformat()
        policy_id = str(data.get("id") or f"{scope_type}:{scope_id}" or uuid.uuid4())
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO quota_policies
                    (id, scope_type, scope_id, name, max_requests_per_day, max_tokens_per_day,
                     max_cost_per_day, max_requests_per_week, max_tokens_per_week, max_cost_per_week,
                     max_requests_per_month, max_tokens_per_month,
                     max_cost_per_month, enabled, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_type, scope_id) DO UPDATE SET
                    name = excluded.name,
                    max_requests_per_day = excluded.max_requests_per_day,
                    max_tokens_per_day = excluded.max_tokens_per_day,
                    max_cost_per_day = excluded.max_cost_per_day,
                    max_requests_per_week = excluded.max_requests_per_week,
                    max_tokens_per_week = excluded.max_tokens_per_week,
                    max_cost_per_week = excluded.max_cost_per_week,
                    max_requests_per_month = excluded.max_requests_per_month,
                    max_tokens_per_month = excluded.max_tokens_per_month,
                    max_cost_per_month = excluded.max_cost_per_month,
                    enabled = excluded.enabled,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
            """, (
                policy_id,
                scope_type,
                scope_id,
                str(data.get("name") or f"{scope_type}:{scope_id}"),
                int(data.get("max_requests_per_day") or 0),
                int(data.get("max_tokens_per_day") or 0),
                float(data.get("max_cost_per_day") or 0),
                int(data.get("max_requests_per_week") or 0),
                int(data.get("max_tokens_per_week") or 0),
                float(data.get("max_cost_per_week") or 0),
                int(data.get("max_requests_per_month") or 0),
                int(data.get("max_tokens_per_month") or 0),
                float(data.get("max_cost_per_month") or 0),
                1 if data.get("enabled", True) else 0,
                json.dumps(data.get("metadata") or {}, ensure_ascii=False),
                now,
                now,
            ))
        return self.get_policy(scope_type=scope_type, scope_id=scope_id) or {}

    def get_policy(self, *, scope_type: str, scope_id: str = "", conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        close = conn is None
        conn = conn or self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM quota_policies WHERE scope_type = ? AND scope_id = ?",
                (scope_type, scope_id),
            ).fetchone()
            return _row_to_dict(row)
        finally:
            if close:
                conn.close()

    def list_policies(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM quota_policies ORDER BY scope_type, scope_id").fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def check_preflight(self, *, user_id: str, org_id: str = "") -> QuotaDecision:
        if not self.enabled:
            return QuotaDecision(allowed=True, reason="quota disabled", user_id=user_id, org_id=org_id, status={"enabled": False})
        status = self.status(user_id=user_id, org_id=org_id)
        exceeded = [item for item in status["checks"] if item["limit"] > 0 and item["used"] >= item["limit"]]
        if exceeded:
            first = exceeded[0]
            return QuotaDecision(
                allowed=False,
                reason=f"Quota exceeded: {first['scope_type']} {first['metric']} {first['used']}/{first['limit']} in {first['window']}",
                user_id=user_id,
                org_id=org_id,
                status=status,
            )
        return QuotaDecision(allowed=True, reason="ok", user_id=user_id, org_id=org_id, status=status)

    def record_usage(self, *, user_id: str, org_id: str = "", total_tokens: int = 0, total_cost: float = 0.0) -> dict[str, Any]:
        if not self.enabled:
            return self.status(user_id=user_id, org_id=org_id)
        now = datetime.now(timezone.utc)
        scopes = [("default", ""), ("user", user_id)]
        if org_id:
            scopes.append(("org", org_id))
        with self._connect() as conn:
            for scope_type, scope_id in scopes:
                for window, window_start in _current_windows(now):
                    conn.execute("""
                        INSERT INTO quota_counters
                            (scope_type, scope_id, window, window_start, request_count, total_tokens, total_cost, updated_at)
                        VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                        ON CONFLICT(scope_type, scope_id, window, window_start) DO UPDATE SET
                            request_count = request_count + 1,
                            total_tokens = total_tokens + excluded.total_tokens,
                            total_cost = total_cost + excluded.total_cost,
                            updated_at = excluded.updated_at
                    """, (scope_type, scope_id, window, window_start, int(total_tokens or 0), float(total_cost or 0), now.isoformat()))
        return self.status(user_id=user_id, org_id=org_id)

    def status(self, *, user_id: str, org_id: str = "") -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        scopes = [("default", ""), ("user", user_id)]
        if org_id:
            scopes.append(("org", org_id))
        with self._connect() as conn:
            policies = [self.get_policy(scope_type=s, scope_id=i, conn=conn) for s, i in scopes]
            counters = {
                (row["scope_type"], row["scope_id"], row["window"]): dict(row)
                for row in conn.execute(
                    """
                    SELECT * FROM quota_counters
                    WHERE (window = 'day' AND window_start = ?)
                       OR (window = 'week' AND window_start = ?)
                       OR (window = 'month' AND window_start = ?)
                    """,
                    (_window_start(now, "day"), _window_start(now, "week"), _window_start(now, "month")),
                ).fetchall()
            }
        checks: list[dict[str, Any]] = []
        for policy in [p for p in policies if p and p.get("enabled")]:
            for window in ("day", "week", "month"):
                counter = counters.get((policy["scope_type"], policy["scope_id"], window), {})
                suffix = f"per_{window}"
                checks.extend([
                    _check(policy, counter, window, "requests", "request_count", f"max_requests_{suffix}"),
                    _check(policy, counter, window, "tokens", "total_tokens", f"max_tokens_{suffix}"),
                    _check(policy, counter, window, "cost", "total_cost", f"max_cost_{suffix}"),
                ])
        return {"enabled": self.enabled, "user_id": user_id, "org_id": org_id, "checks": checks}


def _check(policy: dict[str, Any], counter: dict[str, Any], window: str, metric: str, counter_key: str, limit_key: str) -> dict[str, Any]:
    used = float(counter.get(counter_key) or 0) if metric == "cost" else int(counter.get(counter_key) or 0)
    limit = float(policy.get(limit_key) or 0) if metric == "cost" else int(policy.get(limit_key) or 0)
    return {
        "scope_type": policy["scope_type"],
        "scope_id": policy["scope_id"],
        "window": window,
        "metric": metric,
        "used": used,
        "limit": limit,
        "remaining": None if limit <= 0 else max(0, limit - used),
    }


def _current_windows(now: datetime) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (
        ("day", _window_start(now, "day")),
        ("week", _window_start(now, "week")),
        ("month", _window_start(now, "month")),
    )


def _window_start(now: datetime, window: str) -> str:
    if window == "day":
        return now.strftime("%Y-%m-%d")
    if window == "week":
        monday = (now - timedelta(days=now.weekday())).date()
        return monday.isoformat()
    if window == "month":
        return now.strftime("%Y-%m")
    raise ValueError(f"unknown quota window: {window}")


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if "metadata" in data:
        try:
            data["metadata"] = json.loads(data["metadata"] or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
    if "enabled" in data:
        data["enabled"] = bool(data["enabled"])
    return data


def _env_int(name: str) -> int:
    try:
        return max(0, int(os.getenv(name, "0") or 0))
    except ValueError:
        return 0


def _env_float(name: str) -> float:
    try:
        return max(0.0, float(os.getenv(name, "0") or 0))
    except ValueError:
        return 0.0
