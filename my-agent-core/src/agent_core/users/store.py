from __future__ import annotations

import json
import os
import hashlib
import hmac
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"


class UserStore:
    """SQLite-backed users / organizations store.

    Mirrors Claude Code's account/organization split while remaining generic:
    users own an account_uuid, organizations own an organization_uuid, and
    memberships connect the two.  A deterministic local default user/org keeps
    single-user deployments backward compatible.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = Path(os.getenv("AGENT_USERS_DB_PATH") or _DEFAULT_DB_DIR / "users.db")
        self._db_path = str(Path(db_path).expanduser())
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                TEXT PRIMARY KEY,
                    account_uuid      TEXT NOT NULL UNIQUE,
                    email             TEXT NOT NULL DEFAULT '',
                    name              TEXT NOT NULL DEFAULT '',
                    status            TEXT NOT NULL DEFAULT 'active',
                    password_hash     TEXT NOT NULL DEFAULT '',
                    password_updated_at TEXT,
                    subscription_type TEXT,
                    rate_limit_tier   TEXT,
                    metadata          TEXT NOT NULL DEFAULT '{}',
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL
                )
            """)
            self._ensure_column(conn, "users", "password_hash", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "users", "password_updated_at", "TEXT")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS organizations (
                    id                TEXT PRIMARY KEY,
                    organization_uuid TEXT NOT NULL UNIQUE,
                    name              TEXT NOT NULL DEFAULT '',
                    billing_type      TEXT,
                    status            TEXT NOT NULL DEFAULT 'active',
                    metadata          TEXT NOT NULL DEFAULT '{}',
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS organization_members (
                    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    org_id     TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    role       TEXT NOT NULL DEFAULT 'member',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, org_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_org_members_org ON organization_members(org_id)")
            self.ensure_default_user(conn=conn)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def ensure_default_user(self, *, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
        close = conn is None
        conn = conn or self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            user_id = os.getenv("AGENT_DEFAULT_USER_ID") or "local-user"
            org_id = os.getenv("AGENT_DEFAULT_ORG_ID") or "local-org"
            account_uuid = os.getenv("AGENT_DEFAULT_ACCOUNT_UUID") or "00000000-0000-0000-0000-000000000001"
            organization_uuid = os.getenv("AGENT_DEFAULT_ORGANIZATION_UUID") or "00000000-0000-0000-0000-000000000002"
            conn.execute("""
                INSERT OR IGNORE INTO users
                    (id, account_uuid, email, name, subscription_type, rate_limit_tier, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                account_uuid,
                os.getenv("AGENT_DEFAULT_USER_EMAIL") or "local@my-agent",
                os.getenv("AGENT_DEFAULT_USER_NAME") or "Local User",
                os.getenv("AGENT_DEFAULT_SUBSCRIPTION_TYPE") or "local",
                os.getenv("AGENT_DEFAULT_RATE_LIMIT_TIER") or "default",
                json.dumps({"source": "default"}, ensure_ascii=False),
                now,
                now,
            ))
            conn.execute("""
                INSERT OR IGNORE INTO organizations
                    (id, organization_uuid, name, billing_type, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                org_id,
                organization_uuid,
                os.getenv("AGENT_DEFAULT_ORG_NAME") or "Local Organization",
                os.getenv("AGENT_DEFAULT_BILLING_TYPE") or "local",
                json.dumps({"source": "default"}, ensure_ascii=False),
                now,
                now,
            ))
            conn.execute("""
                INSERT OR IGNORE INTO organization_members (user_id, org_id, role, created_at)
                VALUES (?, ?, 'owner', ?)
            """, (user_id, org_id, now))
            return self.get_user_context(user_id=user_id, org_id=org_id, conn=conn)
        finally:
            if close:
                conn.close()

    def upsert_user(self, data: dict[str, Any]) -> dict[str, Any]:
        user_id = str(data.get("id") or data.get("user_id") or uuid.uuid4()).strip()
        account_uuid = str(data.get("account_uuid") or data.get("accountUuid") or uuid.uuid4()).strip()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO users
                    (id, account_uuid, email, name, status, subscription_type, rate_limit_tier, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    account_uuid = excluded.account_uuid,
                    email = excluded.email,
                    name = excluded.name,
                    status = excluded.status,
                    subscription_type = excluded.subscription_type,
                    rate_limit_tier = excluded.rate_limit_tier,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
            """, (
                user_id,
                account_uuid,
                str(data.get("email") or ""),
                str(data.get("name") or ""),
                str(data.get("status") or "active"),
                data.get("subscription_type") or data.get("subscriptionType"),
                data.get("rate_limit_tier") or data.get("rateLimitTier"),
                json.dumps(data.get("metadata") or {}, ensure_ascii=False),
                now,
                now,
            ))
            password = data.get("password") or data.get("plain_password") or data.get("plainPassword")
            if password:
                self.set_password(user_id=user_id, password=str(password), conn=conn)
        user = self.get_user(user_id)
        if user is None:
            raise RuntimeError("failed to upsert user")
        return user

    def upsert_organization(self, data: dict[str, Any]) -> dict[str, Any]:
        org_id = str(data.get("id") or data.get("org_id") or uuid.uuid4()).strip()
        organization_uuid = str(data.get("organization_uuid") or data.get("organizationUuid") or uuid.uuid4()).strip()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO organizations
                    (id, organization_uuid, name, billing_type, status, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    organization_uuid = excluded.organization_uuid,
                    name = excluded.name,
                    billing_type = excluded.billing_type,
                    status = excluded.status,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
            """, (
                org_id,
                organization_uuid,
                str(data.get("name") or ""),
                data.get("billing_type") or data.get("billingType"),
                str(data.get("status") or "active"),
                json.dumps(data.get("metadata") or {}, ensure_ascii=False),
                now,
                now,
            ))
        org = self.get_organization(org_id)
        if org is None:
            raise RuntimeError("failed to upsert organization")
        return org

    def add_member(self, *, user_id: str, org_id: str, role: str = "member") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO organization_members (user_id, org_id, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, org_id) DO UPDATE SET role = excluded.role
            """, (user_id, org_id, role, now))

    def set_password(self, *, user_id: str, password: str, conn: sqlite3.Connection | None = None) -> None:
        if len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        close = conn is None
        conn = conn or self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE users SET password_hash = ?, password_updated_at = ?, updated_at = ? WHERE id = ?",
                (_hash_password(password), now, now, user_id),
            )
        finally:
            if close:
                conn.close()

    def verify_password(self, *, user_id: str, password: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None or not row["password_hash"]:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        user = _row_to_dict(row)
        if user and user.get("status") != "active":
            return None
        return user

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_dict(row)

    def get_organization(self, org_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM organizations WHERE id = ?", (org_id,)).fetchone()
        return _row_to_dict(row)

    def get_user_context(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        user_id = user_id or os.getenv("AGENT_DEFAULT_USER_ID") or "local-user"
        org_id = org_id or os.getenv("AGENT_DEFAULT_ORG_ID") or "local-org"
        close = conn is None
        conn = conn or self._connect()
        try:
            user = _row_to_dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
            org = _row_to_dict(conn.execute("SELECT * FROM organizations WHERE id = ?", (org_id,)).fetchone())
            if user is None or org is None:
                return self.ensure_default_user(conn=conn)
            member = conn.execute(
                "SELECT role FROM organization_members WHERE user_id = ? AND org_id = ?",
                (user_id, org_id),
            ).fetchone()
            return {"user": user, "organization": org, "role": member["role"] if member else "member"}
        finally:
            if close:
                conn.close()

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]

    def list_organizations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM organizations ORDER BY created_at DESC").fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("metadata",):
        if key in data:
            try:
                data[key] = json.loads(data[key] or "{}")
            except json.JSONDecodeError:
                data[key] = {}
    data["has_password"] = bool(data.get("password_hash"))
    data.pop("password_hash", None)
    return data


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 240_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_raw, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False
