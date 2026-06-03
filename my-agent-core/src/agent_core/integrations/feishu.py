from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_core.sqlite_utils import connect_sqlite
from agent_core.tools.base import ToolContext, ToolResult


_DEFAULT_DB_DIR = Path.home() / ".my-agent-core"
_DEFAULT_FEISHU_BASE_URL = "https://open.feishu.cn"
_MAX_FEISHU_RESPONSE_BYTES = 5 * 1024 * 1024
_FEISHU_TIMEOUT_SECONDS = 30


FeishuRequester = Callable[[str, str, dict[str, str], bytes | None], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class FeishuConnectionStatus:
    connected: bool
    user_id: str
    org_id: str
    credential_type: str
    scopes: list[str]
    expires_at: float | None
    token_type: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "credential_type": self.credential_type,
            "scopes": self.scopes,
            "expires_at": self.expires_at,
            "expires_at_iso": _iso_from_ts(self.expires_at),
            "token_type": self.token_type,
            "metadata": self.metadata,
        }


class FeishuTokenStore:
    """SQLite-backed per-user Feishu token store.

    Tokens are keyed by our core user_id + org_id instead of being global bot
    credentials.  This keeps Feishu access scoped to the authenticated terminal
    user and lets each user connect/revoke their own Feishu account.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = Path(os.getenv("AGENT_FEISHU_DB_PATH") or _DEFAULT_DB_DIR / "feishu.db")
        self._db_path = str(Path(db_path).expanduser())
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, row_factory=sqlite3.Row, pragmas=("PRAGMA journal_mode=WAL",))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_user_tokens (
                    user_id       TEXT NOT NULL,
                    org_id        TEXT NOT NULL DEFAULT '',
                    access_token  TEXT NOT NULL,
                    refresh_token TEXT NOT NULL DEFAULT '',
                    credential_type TEXT NOT NULL DEFAULT 'user_access_token',
                    app_id        TEXT NOT NULL DEFAULT '',
                    app_secret    TEXT NOT NULL DEFAULT '',
                    token_type    TEXT NOT NULL DEFAULT 'Bearer',
                    scopes        TEXT NOT NULL DEFAULT '[]',
                    expires_at    REAL,
                    metadata      TEXT NOT NULL DEFAULT '{}',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (user_id, org_id)
                )
                """
            )
            self._ensure_column(conn, "feishu_user_tokens", "credential_type", "TEXT NOT NULL DEFAULT 'user_access_token'")
            self._ensure_column(conn, "feishu_user_tokens", "app_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "feishu_user_tokens", "app_secret", "TEXT NOT NULL DEFAULT ''")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def save_user_token(
        self,
        *,
        user_id: str,
        org_id: str = "",
        access_token: str,
        refresh_token: str = "",
        expires_in: int | float | None = None,
        expires_at: int | float | None = None,
        scopes: list[str] | None = None,
        token_type: str = "Bearer",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _require_non_empty("user_id", user_id)
        org_id = str(org_id or "")
        access_token = _require_non_empty("access_token", access_token)
        expiry = _resolve_expiry(expires_in=expires_in, expires_at=expires_at)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feishu_user_tokens
                    (user_id, org_id, access_token, refresh_token, credential_type, app_id, app_secret, token_type, scopes, expires_at, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, org_id) DO UPDATE SET
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    credential_type = excluded.credential_type,
                    app_id = excluded.app_id,
                    app_secret = excluded.app_secret,
                    token_type = excluded.token_type,
                    scopes = excluded.scopes,
                    expires_at = excluded.expires_at,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    org_id,
                    access_token,
                    str(refresh_token or ""),
                    "user_access_token",
                    "",
                    "",
                    str(token_type or "Bearer"),
                    json.dumps(scopes or [], ensure_ascii=False),
                    expiry,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        token = self.get_user_token(user_id=user_id, org_id=org_id, include_secret=False)
        if token is None:
            raise RuntimeError("failed to save Feishu token")
        return token

    def save_app_credentials(
        self,
        *,
        user_id: str,
        org_id: str = "",
        app_id: str,
        app_secret: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _require_non_empty("user_id", user_id)
        org_id = str(org_id or "")
        app_id = _require_non_empty("app_id", app_id)
        app_secret = _require_non_empty("app_secret", app_secret)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feishu_user_tokens
                    (user_id, org_id, access_token, refresh_token, credential_type, app_id, app_secret, token_type, scopes, expires_at, metadata, created_at, updated_at)
                VALUES (?, ?, '', '', 'app_credentials', ?, ?, 'Bearer', '[]', NULL, ?, ?, ?)
                ON CONFLICT(user_id, org_id) DO UPDATE SET
                    access_token = '',
                    refresh_token = '',
                    credential_type = excluded.credential_type,
                    app_id = excluded.app_id,
                    app_secret = excluded.app_secret,
                    token_type = excluded.token_type,
                    scopes = excluded.scopes,
                    expires_at = excluded.expires_at,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    org_id,
                    app_id,
                    app_secret,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        token = self.get_user_token(user_id=user_id, org_id=org_id, include_secret=False)
        if token is None:
            raise RuntimeError("failed to save Feishu app credentials")
        return token

    def get_user_token(
        self,
        *,
        user_id: str,
        org_id: str = "",
        include_secret: bool = False,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM feishu_user_tokens WHERE user_id = ? AND org_id = ?",
                (str(user_id or ""), str(org_id or "")),
            ).fetchone()
        return _token_row_to_dict(row, include_secret=include_secret)

    def update_metadata(self, *, user_id: str, org_id: str = "", metadata: dict[str, Any]) -> dict[str, Any] | None:
        token = self.get_user_token(user_id=user_id, org_id=org_id, include_secret=False)
        if token is None:
            return None
        merged = {**dict(token.get("metadata") or {}), **dict(metadata or {})}
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE feishu_user_tokens SET metadata = ?, updated_at = ? WHERE user_id = ? AND org_id = ?",
                (json.dumps(merged, ensure_ascii=False), now, str(user_id or ""), str(org_id or "")),
            )
        return self.get_user_token(user_id=user_id, org_id=org_id, include_secret=False)

    def list_tokens(self, *, include_secret: bool = False) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feishu_user_tokens ORDER BY updated_at DESC").fetchall()
        return [_token_row_to_dict(row, include_secret=include_secret) for row in rows if row is not None]

    def delete_user_token(self, *, user_id: str, org_id: str = "") -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM feishu_user_tokens WHERE user_id = ? AND org_id = ?",
                (str(user_id or ""), str(org_id or "")),
            )
        return cur.rowcount > 0

    def status(self, *, user_id: str, org_id: str = "") -> FeishuConnectionStatus:
        token = self.get_user_token(user_id=user_id, org_id=org_id, include_secret=False)
        if token is None:
            return FeishuConnectionStatus(
                connected=False,
                user_id=str(user_id or ""),
                org_id=str(org_id or ""),
                credential_type="none",
                scopes=[],
                expires_at=None,
                token_type="Bearer",
                metadata={},
            )
        return FeishuConnectionStatus(
            connected=not bool(token.get("expired")),
            user_id=token["user_id"],
            org_id=token["org_id"],
            credential_type=str(token.get("credential_type") or "user_access_token"),
            scopes=list(token.get("scopes") or []),
            expires_at=token.get("expires_at"),
            token_type=str(token.get("token_type") or "Bearer"),
            metadata=dict(token.get("metadata") or {}),
        )


class FeishuApiTool:
    name = "FeishuApi"
    description = """Call Feishu/Lark OpenAPI with the current MyAgent user's stored Feishu credentials.

Use this for authenticated Feishu resources after the user has connected Feishu via
/terminal/api/integrations/feishu. Credentials are resolved by user_id + org_id
from runtime metadata. The preferred OpenClaw-compatible mode is app credentials
(App ID + App Secret), which are exchanged for a tenant access token automatically.
User Access Token mode is still supported for user-scoped APIs.

Inputs:
- method: GET, POST, PUT, PATCH, DELETE
- path: OpenAPI path, e.g. /open-apis/authen/v1/user_info
- query: optional object of query parameters
- body: optional JSON object/array/string for non-GET requests
"""
    input_schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
            "path": {"type": "string", "description": "Feishu OpenAPI path beginning with /open-apis/"},
            "query": {"type": "object", "description": "Optional query parameters"},
            "body": {"description": "Optional JSON body"},
        },
        "required": ["method", "path"],
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(
        self,
        token_store: FeishuTokenStore | None = None,
        *,
        base_url: str | None = None,
        requester: FeishuRequester | None = None,
    ) -> None:
        self._token_store = token_store or FeishuTokenStore()
        self._base_url = (base_url or os.getenv("AGENT_FEISHU_BASE_URL") or _DEFAULT_FEISHU_BASE_URL).rstrip("/")
        self._requester = requester

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        user_id = str(context.metadata.get("user_id") or "").strip()
        org_id = str(context.metadata.get("org_id") or "").strip()
        if not user_id:
            return ToolResult(content="FeishuApi failed: runtime has no user_id metadata", is_error=True)
        token = self._token_store.get_user_token(user_id=user_id, org_id=org_id, include_secret=True)
        if not token:
            return ToolResult(
                content=(
                    "FeishuApi failed: current user has not connected Feishu. "
                    "Connect Feishu through /terminal/api/integrations/feishu."
                ),
                is_error=True,
                metadata={"user_id": user_id, "org_id": org_id, "connected": False},
            )
        credential_type = str(token.get("credential_type") or "user_access_token")
        if credential_type == "user_access_token" and token.get("expired"):
            return ToolResult(
                content="FeishuApi failed: current user's Feishu token is expired; reconnect or refresh it before retrying.",
                is_error=True,
                metadata={"user_id": user_id, "org_id": org_id, "connected": True, "expired": True},
            )

        method = str(tool_input.get("method") or "GET").strip().upper()
        path = str(tool_input.get("path") or "").strip()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return ToolResult(content=f"FeishuApi failed: unsupported method {method!r}", is_error=True)
        if not path.startswith("/open-apis/"):
            return ToolResult(content="FeishuApi failed: path must start with /open-apis/", is_error=True)

        url = _build_url(self._base_url, path, tool_input.get("query"))
        body_bytes = _encode_body(tool_input.get("body"), method)
        access_token = token.get("access_token")
        if credential_type == "app_credentials":
            try:
                access_token = await self._tenant_access_token(
                    app_id=str(token.get("app_id") or ""),
                    app_secret=str(token.get("app_secret") or ""),
                )
            except Exception as exc:
                return ToolResult(
                    content=f"FeishuApi failed to exchange app credentials for tenant access token: {exc}",
                    is_error=True,
                    metadata={"method": method, "path": path, "user_id": user_id, "org_id": org_id},
                )

        headers = {
            "Authorization": f"{token.get('token_type') or 'Bearer'} {access_token}",
            "User-Agent": "my-agent-core-feishu/0.1",
            "Accept": "application/json",
        }
        if body_bytes is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"

        try:
            payload = await self._request(method, url, headers, body_bytes)
        except Exception as exc:
            return ToolResult(
                content=f"FeishuApi failed for {method} {path}: {exc}",
                is_error=True,
                metadata={"method": method, "path": path, "user_id": user_id, "org_id": org_id},
            )

        safe_payload = _redact_secrets(payload)
        return ToolResult(
            content=json.dumps(safe_payload, ensure_ascii=False, indent=2),
            metadata={"method": method, "path": path, "user_id": user_id, "org_id": org_id},
        )

    async def _request(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
        if self._requester:
            return await self._requester(method, url, headers, body)
        return await asyncio.to_thread(_urllib_json_request, method, url, headers, body)

    async def _tenant_access_token(self, *, app_id: str, app_secret: str) -> str:
        app_id = _require_non_empty("app_id", app_id)
        app_secret = _require_non_empty("app_secret", app_secret)
        url = f"{self._base_url}/open-apis/auth/v3/tenant_access_token/internal"
        payload = await self._request(
            "POST",
            url,
            {
                "User-Agent": "my-agent-core-feishu/0.1",
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            },
            json.dumps({"app_id": app_id, "app_secret": app_secret}, ensure_ascii=False).encode("utf-8"),
        )
        if payload.get("code") not in (None, 0):
            raise RuntimeError(json.dumps(_redact_secrets(payload), ensure_ascii=False))
        tenant_access_token = str(payload.get("tenant_access_token") or "").strip()
        if not tenant_access_token:
            raise RuntimeError("tenant_access_token missing from Feishu response")
        return tenant_access_token


def _urllib_json_request(method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_FEISHU_TIMEOUT_SECONDS) as resp:
            raw = resp.read(_MAX_FEISHU_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_FEISHU_RESPONSE_BYTES:
                raise ValueError("response too large")
            text = raw.decode("utf-8", errors="replace")
            return _parse_json_response(text, status=resp.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read(_MAX_FEISHU_RESPONSE_BYTES + 1)
        text = raw.decode("utf-8", errors="replace")
        parsed = _parse_json_response(text, status=exc.code)
        raise RuntimeError(json.dumps(_redact_secrets(parsed), ensure_ascii=False)) from exc


def _parse_json_response(text: str, *, status: int) -> dict[str, Any]:
    if not text.strip():
        return {"status": status, "data": None}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"status": status, "text": text}
    if isinstance(parsed, dict):
        return {"status": status, **parsed}
    return {"status": status, "data": parsed}


def _build_url(base_url: str, path: str, query: Any) -> str:
    url = f"{base_url}{path}"
    if not isinstance(query, dict) or not query:
        return url
    pairs: list[tuple[str, str]] = []
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, list):
            pairs.extend((str(key), str(item)) for item in value)
        else:
            pairs.append((str(key), str(value)))
    encoded = urllib.parse.urlencode(pairs)
    return f"{url}?{encoded}" if encoded else url


def _encode_body(body: Any, method: str) -> bytes | None:
    if method == "GET" and body in (None, "", {}, []):
        return None
    if body is None:
        return None
    if isinstance(body, str):
        return body.encode("utf-8")
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _resolve_expiry(*, expires_in: int | float | None, expires_at: int | float | None) -> float | None:
    if expires_at is not None:
        return float(expires_at)
    if expires_in is None:
        return None
    return time.time() + float(expires_in)


def _token_row_to_dict(row: sqlite3.Row | None, *, include_secret: bool) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key, fallback in (("scopes", []), ("metadata", {})):
        try:
            data[key] = json.loads(data.get(key) or json.dumps(fallback))
        except json.JSONDecodeError:
            data[key] = fallback
    expires_at = data.get("expires_at")
    data["expired"] = bool(expires_at is not None and float(expires_at) <= time.time())
    data["expires_at_iso"] = _iso_from_ts(expires_at)
    data["has_refresh_token"] = bool(data.get("refresh_token"))
    if not include_secret:
        data.pop("access_token", None)
        data.pop("refresh_token", None)
        data.pop("app_secret", None)
    return data


def _iso_from_ts(ts: int | float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()


def _require_non_empty(name: str, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(marker in lower for marker in ("token", "secret", "password", "authorization")):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value
