from __future__ import annotations

import asyncio
import contextlib
from email import policy
from email.parser import BytesParser
import json
import os
import re
import signal
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response as FastAPIResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent_core.adapters.web import agent_sse, event_to_chat_json
from agent_core.billing.store import BillingStore
from agent_core.buddy import companion_payload, get_companion
from agent_core.core.events import AgentEvent
from agent_core.integrations.feishu import FeishuApiTool, FeishuTokenStore
from agent_core.integrations.feishu_ws_bridge import FeishuWebSocketBridge
from agent_core.plan_mode import export_plan_metadata, get_plan
from agent_core.quota.store import QuotaStore
from agent_core.session.store import create_session_store, deserialize_message
from agent_core.tools.base import ToolContext
from agent_core.users.store import UserStore
from agent_server.admin import (
    _ADMIN_COOKIE_NAME,
    get_admin_token,
    is_admin_authenticated,
    print_generated_admin_token_once,
    router as admin_router,
)
from agent_core.images import ImageInput
from agent_server.runtime_factory import clear_image_generation_client_cache, create_runtime, create_worktree_manager, get_attachment_store, get_image_generation_client, get_task_store
from agent_server.runtime_factory import default_prompt_template_name, get_session_prompt_template, list_prompt_templates
from agent_server.searxng_service import searxng_service
from agent_server.slash_commands import command_specs, handle_slash_command
from agent_server.task_manager import agent_task_manager
from agent_server.codex_credentials import codex_credential_source, codex_login_status, delete_codex_credentials, save_codex_credentials, start_codex_login_flow
from agent_server.model_providers import env_model_selection

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _content_disposition_inline(filename: str) -> str:
    """Build a latin-1-safe Content-Disposition value for arbitrary filenames."""
    basename = Path(filename or "attachment").name or "attachment"
    ascii_fallback = basename.encode("ascii", "ignore").decode("ascii").strip() or "attachment"
    if ascii_fallback.startswith("."):
        ascii_suffix = Path(basename).suffix.encode("ascii", "ignore").decode("ascii")
        ascii_fallback = f"attachment{ascii_suffix}"
    ascii_fallback = ascii_fallback.replace("\\", "\\\\").replace('"', r"\"")
    encoded = quote(basename.encode("utf-8"), safe="")
    return f'inline; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str | None = Field(
        default=None,
        description="模型 provider；不传时读取 AGENT_MODEL_PROVIDER/ProviderSpec 自动探测。",
    )
    session_id: str | None = None


class AdminLoginRequest(BaseModel):
    token: str = Field(..., min_length=1)


class TerminalLoginRequest(BaseModel):
    token: str = ""
    user_id: str | None = None
    org_id: str | None = None
    password: str = ""


class CreateTerminalSessionRequest(BaseModel):
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class TerminalSessionConfigRequest(BaseModel):
    prompt_template: str | None = None
    title: str | None = None


class TerminalAttachmentRef(BaseModel):
    id: str
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None


class ImagePlaygroundRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="generate", pattern="^(generate|edit)$")
    size: str = "1024x1024"
    quality: str | None = None
    output_format: str = "png"
    n: int = Field(default=1, ge=1, le=10)
    input_attachment_ids: list[str] = Field(default_factory=list)
    mask_attachment_id: str | None = None


def _parse_multipart_file(body: bytes, content_type: str, *, field_name: str = "file") -> tuple[str, str | None, bytes]:
    """Parse a single multipart file without depending on python-multipart.

    Starlette's ``request.form()`` requires the optional python-multipart
    package.  The terminal image upload should still work in lightweight dev
    installs, so keep a narrow stdlib fallback for the one file field we need.
    """
    if not body:
        raise ValueError("request body is empty")
    if "multipart/form-data" not in (content_type or "").lower():
        raise ValueError("request must be multipart/form-data")
    raw = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not message.is_multipart():
        raise ValueError("invalid multipart body")
    for part in message.iter_parts():
        params = dict(part.get_params(header="content-disposition") or [])
        if params.get("name") != field_name:
            continue
        data = part.get_payload(decode=True) or b""
        filename = str(params.get("filename") or "image")
        return filename, part.get_content_type(), data
    raise ValueError(f"{field_name} field is required")


class TaskAssignRequest(BaseModel):
    owner: str = Field(..., min_length=1)


class FeishuTokenRequest(BaseModel):
    access_token: str = Field(..., min_length=1)
    refresh_token: str = ""
    expires_in: int | None = Field(default=None, ge=1)
    expires_at: float | None = None
    scopes: list[str] = Field(default_factory=list)
    token_type: str = "Bearer"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeishuAppCredentialsRequest(BaseModel):
    credential: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodexCredentialsRequest(BaseModel):
    secret: str = Field(..., min_length=1)
    credential_type: str = Field(default="api_key", pattern="^(api_key|access_token|auth_json)$")
    base_url: str = "http://localhost:8080/v1"
    model: str = "gpt-5.4"
    image_model: str = "gpt-image-2"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodexLoginRequest(BaseModel):
    flow: str = Field(default="browser", pattern="^(browser|device)$")


_TERMINAL_COOKIE_NAME = "my_agent_terminal_token"
_TERMINAL_USER_COOKIE_NAME = "my_agent_user_id"
_TERMINAL_ORG_COOKIE_NAME = "my_agent_org_id"
_GENERATED_TERMINAL_TOKEN = secrets.token_urlsafe(32)
_GENERATED_TERMINAL_TOKEN_PRINTED = False


app = FastAPI(title="My Agent Core Debug Server", version="0.1.0")
app.include_router(admin_router)
_billing_store = BillingStore()
_user_store = UserStore()
_quota_store = QuotaStore()
_feishu_token_store = FeishuTokenStore()
_feishu_ws_bridge = FeishuWebSocketBridge(token_store=_feishu_token_store, runtime_factory=create_runtime)

print_generated_admin_token_once()


def get_terminal_token() -> str:
    return (
        os.getenv("AGENT_TERMINAL_TOKEN")
        or os.getenv("TERMINAL_TOKEN")
        or os.getenv("AGENT_TERMINAL_KEY")
        or _GENERATED_TERMINAL_TOKEN
    )


def print_generated_terminal_token_once() -> None:
    global _GENERATED_TERMINAL_TOKEN_PRINTED
    if _GENERATED_TERMINAL_TOKEN_PRINTED:
        return
    if os.getenv("AGENT_TERMINAL_TOKEN") or os.getenv("TERMINAL_TOKEN") or os.getenv("AGENT_TERMINAL_KEY"):
        return
    _GENERATED_TERMINAL_TOKEN_PRINTED = True
    print(
        "[terminal] AGENT_TERMINAL_TOKEN is not set. Generated temporary terminal token for this process:\n"
        f"[terminal] {_GENERATED_TERMINAL_TOKEN}\n"
        "[terminal] Set AGENT_TERMINAL_TOKEN to a stable secret for persistent access.",
        file=sys.stderr,
    )


print_generated_terminal_token_once()


def _extract_terminal_token(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-terminal-token")
        or request.cookies.get(_TERMINAL_COOKIE_NAME)
        or request.query_params.get("terminal_token")
        or ""
    )


def is_terminal_authenticated(request: Request) -> bool:
    supplied = _extract_terminal_token(request)
    expected = get_terminal_token()
    return bool(supplied) and secrets.compare_digest(supplied, expected)


def is_terminal_ws_authenticated(websocket: WebSocket) -> bool:
    auth = websocket.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    else:
        supplied = (
            websocket.headers.get("x-terminal-token")
            or websocket.cookies.get(_TERMINAL_COOKIE_NAME)
            or websocket.query_params.get("terminal_token")
            or ""
        )
    return bool(supplied) and secrets.compare_digest(supplied, get_terminal_token())


def _resolve_user_context(user_id: str | None = None, org_id: str | None = None) -> dict[str, Any]:
    return _user_store.get_user_context(user_id=user_id or None, org_id=org_id or None)


def get_default_user_context(request: Request | None = None, websocket: WebSocket | None = None) -> dict[str, Any]:
    user_id = org_id = None
    if request is not None:
        user_id = request.headers.get("x-agent-user-id") or request.cookies.get(_TERMINAL_USER_COOKIE_NAME)
        org_id = request.headers.get("x-agent-org-id") or request.cookies.get(_TERMINAL_ORG_COOKIE_NAME)
    if websocket is not None:
        user_id = websocket.headers.get("x-agent-user-id") or websocket.cookies.get(_TERMINAL_USER_COOKIE_NAME) or websocket.query_params.get("user_id")
        org_id = websocket.headers.get("x-agent-org-id") or websocket.cookies.get(_TERMINAL_ORG_COOKIE_NAME) or websocket.query_params.get("org_id")
    return _resolve_user_context(user_id=user_id, org_id=org_id)


def _parse_feishu_app_credential(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Feishu App credential is required")
    app_id = app_secret = ""
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid Feishu credential JSON: {exc.msg}") from exc
        if isinstance(data, dict):
            app_id = str(data.get("app_id") or data.get("appId") or "").strip()
            app_secret = str(data.get("app_secret") or data.get("appSecret") or "").strip()
    elif ":" in text and "\n" not in text:
        app_id, app_secret = [part.strip() for part in text.split(":", 1)]
    else:
        values: dict[str, str] = {}
        bare_lines: list[str] = []
        for line in text.splitlines():
            line = line.strip().strip(",")
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip().strip('"\'')
            elif ":" in line:
                key, value = line.split(":", 1)
                values[key.strip().lower()] = value.strip().strip('"\'')
            else:
                bare_lines.append(line.strip().strip('"\''))
        app_id = values.get("app_id") or values.get("appid") or values.get("appid") or ""
        app_secret = values.get("app_secret") or values.get("appsecret") or values.get("secret") or ""
        if not app_id and len(bare_lines) >= 1:
            app_id = bare_lines[0]
        if not app_secret and len(bare_lines) >= 2:
            app_secret = bare_lines[1]
    if not app_id or not app_secret:
        raise HTTPException(
            status_code=400,
            detail="请粘贴 App ID + App Secret，支持 JSON、app_id:app_secret、两行或 key=value 格式",
        )
    return {"app_id": app_id, "app_secret": app_secret}


def _assert_session_access(session_info: dict[str, Any] | None, user_ctx: dict[str, Any]) -> None:
    """Enforce terminal session isolation by user_id.

    Admin APIs can still see all sessions. Terminal/chat routes must only touch
    sessions owned by the logged-in user. Legacy sessions with empty user_id are
    treated as unowned and are not accessible from user login sessions.
    """
    if session_info is None:
        return
    owner = str(session_info.get("user_id") or "")
    current = str(user_ctx["user"]["id"])
    if owner != current:
        raise HTTPException(status_code=403, detail="Session does not belong to current user")


def _assert_ws_session_access(session_info: dict[str, Any] | None, user_ctx: dict[str, Any]) -> bool:
    if session_info is None:
        return True
    return str(session_info.get("user_id") or "") == str(user_ctx["user"]["id"])


async def require_terminal_auth(request: Request) -> None:
    if not is_terminal_authenticated(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Terminal authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.on_event("startup")
async def startup_bundled_services() -> None:
    status_info = await searxng_service.start()
    if status_info.enabled and not status_info.running:
        print(f"[searxng] not running: {status_info.reason}", file=sys.stderr)
    elif status_info.running:
        print(f"[searxng] running at {status_info.endpoint} (pid={status_info.pid})", file=sys.stderr)
    _autostart_feishu_bridges()


@app.on_event("shutdown")
async def shutdown_agent_tasks() -> None:
    _stop_feishu_bridges()
    await agent_task_manager.shutdown()
    await searxng_service.stop()


def _autostart_feishu_bridges() -> None:
    """Restart Feishu message bridges that were enabled before a dev reload.

    The websocket bridge is an in-process background thread, so uvicorn reloads
    clear it. Persist only the user's intent in token metadata and recreate the
    bridge at process startup to avoid silently losing Feishu messages.
    """
    for token in _feishu_token_store.list_tokens(include_secret=False):
        metadata = dict(token.get("metadata") or {})
        if token.get("credential_type") != "app_credentials" or not metadata.get("bridge_autostart"):
            continue
        user_id = str(token.get("user_id") or "")
        org_id = str(token.get("org_id") or "")
        if not user_id:
            continue
        try:
            bridge = _feishu_ws_bridge.start(user_id=user_id, org_id=org_id)
            print(f"[feishu] bridge autostarted for {user_id}/{org_id}: {bridge.get('state')}", file=sys.stderr)
        except Exception as exc:
            print(f"[feishu] bridge autostart failed for {user_id}/{org_id}: {exc}", file=sys.stderr)


def _stop_feishu_bridges() -> None:
    for token in _feishu_token_store.list_tokens(include_secret=False):
        metadata = dict(token.get("metadata") or {})
        if token.get("credential_type") != "app_credentials" or not metadata.get("bridge_autostart"):
            continue
        user_id = str(token.get("user_id") or "")
        org_id = str(token.get("org_id") or "")
        if user_id:
            with contextlib.suppress(Exception):
                _feishu_ws_bridge.stop(user_id=user_id, org_id=org_id)


# ── HTTP routes ──────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/searxng")
async def searxng_health() -> dict[str, Any]:
    status_info = searxng_service.status()
    return {
        "enabled": status_info.enabled,
        "running": status_info.running,
        "endpoint": status_info.endpoint,
        "source_dir": status_info.source_dir,
        "settings_path": status_info.settings_path,
        "pid": status_info.pid,
        "reason": status_info.reason,
    }


@app.get("/terminal", response_class=HTMLResponse)
async def terminal_page(request: Request):
    if not is_terminal_authenticated(request):
        return RedirectResponse("/terminal/login", status_code=303)
    response = HTMLResponse((_STATIC_DIR / "terminal.html").read_text(encoding="utf-8"))
    query_token = request.query_params.get("terminal_token")
    if query_token and secrets.compare_digest(query_token, get_terminal_token()):
        _set_terminal_cookie(response, query_token)
    return response


@app.get("/terminal/login", response_class=HTMLResponse)
async def terminal_login_page(request: Request):
    if is_terminal_authenticated(request):
        return RedirectResponse("/terminal", status_code=303)
    return HTMLResponse(_terminal_login_html())


@app.post("/terminal/login")
async def terminal_login(request: TerminalLoginRequest, response: Response) -> dict[str, str]:
    password_login = bool(request.user_id and request.password)
    token_login = bool(request.token and secrets.compare_digest(request.token, get_terminal_token()))
    if password_login:
        user = _user_store.verify_password(user_id=request.user_id or "", password=request.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid user ID or password")
        user_ctx = _resolve_user_context(user_id=user["id"], org_id=request.org_id)
        _set_terminal_cookie(response, get_terminal_token())
    elif token_login:
        user_ctx = _resolve_user_context(user_id=request.user_id, org_id=request.org_id)
        _set_terminal_cookie(response, request.token)
    else:
        raise HTTPException(status_code=401, detail="Invalid login credentials")
    _set_terminal_identity_cookies(response, user_ctx["user"]["id"], user_ctx["organization"]["id"])
    return {"status": "ok", "user_id": user_ctx["user"]["id"], "org_id": user_ctx["organization"]["id"]}


@app.post("/terminal/logout")
async def terminal_logout(response: Response) -> dict[str, str]:
    response.delete_cookie(_TERMINAL_COOKIE_NAME, path="/")
    response.delete_cookie(_TERMINAL_USER_COOKIE_NAME, path="/")
    response.delete_cookie(_TERMINAL_ORG_COOKIE_NAME, path="/")
    return {"status": "logged_out"}


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    if not is_admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=303)
    response = HTMLResponse((_STATIC_DIR / "admin.html").read_text(encoding="utf-8"))
    query_token = request.query_params.get("admin_token")
    if query_token and secrets.compare_digest(query_token, get_admin_token()):
        _set_admin_cookie(response, query_token)
    return response


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    if is_admin_authenticated(request):
        return RedirectResponse("/admin", status_code=303)
    return HTMLResponse(_admin_login_html())


@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest, response: Response) -> dict[str, str]:
    if not request.token or not secrets.compare_digest(request.token, get_admin_token()):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    _set_admin_cookie(response, request.token)
    return {"status": "ok"}


@app.post("/admin/logout")
async def admin_logout(response: Response) -> dict[str, str]:
    response.delete_cookie(_ADMIN_COOKIE_NAME, path="/admin")
    return {"status": "logged_out"}


def _set_admin_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        _ADMIN_COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        secure=os.getenv("AGENT_ADMIN_COOKIE_SECURE", "").lower() in {"1", "true", "yes", "on"},
        max_age=int(os.getenv("AGENT_ADMIN_COOKIE_MAX_AGE", "86400")),
        path="/admin",
    )


def _set_terminal_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        _TERMINAL_COOKIE_NAME,
        token,
        httponly=True,
        samesite="strict",
        secure=os.getenv("AGENT_TERMINAL_COOKIE_SECURE", "").lower() in {"1", "true", "yes", "on"},
        max_age=int(os.getenv("AGENT_TERMINAL_COOKIE_MAX_AGE", "86400")),
        path="/",
    )


def _set_terminal_identity_cookies(response: Response, user_id: str, org_id: str) -> None:
    secure = os.getenv("AGENT_TERMINAL_COOKIE_SECURE", "").lower() in {"1", "true", "yes", "on"}
    max_age = int(os.getenv("AGENT_TERMINAL_COOKIE_MAX_AGE", "86400"))
    for name, value in ((_TERMINAL_USER_COOKIE_NAME, user_id), (_TERMINAL_ORG_COOKIE_NAME, org_id)):
        response.set_cookie(
            name,
            value,
            httponly=True,
            samesite="strict",
            secure=secure,
            max_age=max_age,
            path="/",
        )


def _terminal_login_html() -> str:
    return _token_login_html(
        title="MyAgent Terminal Login",
        heading="终端访问验证",
        description="使用管理员为你创建的用户 ID 和口令登录；服务访问口令仅作为开发/运维兼容入口。",
        endpoint="/terminal/login",
        redirect_to="/terminal",
        input_placeholder="请输入访问口令",
        show_identity=True,
    )


def _admin_login_html() -> str:
    return _token_login_html(
        title="MyAgent Admin Login",
        heading="管理后台验证",
        description="请输入管理员口令以进入管理后台。",
        endpoint="/admin/login",
        redirect_to="/admin",
        input_placeholder="请输入管理员口令",
        show_identity=False,
    )


def _token_login_html(*, title: str, heading: str, description: str, endpoint: str, redirect_to: str, input_placeholder: str, show_identity: bool = False) -> str:
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; display:grid; place-items:center; background:#09090b; color:#e4e4e7; font-family:Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
    .card { width:min(420px, calc(100vw - 32px)); background:#18181b; border:1px solid #27272a; border-radius:14px; padding:24px; box-shadow:0 24px 80px rgba(0,0,0,.35); }
    h1 { font-size:18px; margin:0 0 8px; }
    p { color:#a1a1aa; font-size:13px; line-height:1.5; margin:0 0 18px; }
    input { width:100%; padding:10px 12px; border:1px solid #3f3f46; border-radius:8px; background:#09090b; color:#e4e4e7; outline:none; }
    .identity-fields { display:__IDENTITY_DISPLAY__; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
    .identity-fields input { font-size:12px; }
    .identity-fields .wide { grid-column: 1 / -1; }
    .login-note { display:__IDENTITY_DISPLAY__; color:#71717a; font-size:11px; line-height:1.5; margin:10px 0 0; }
    button { width:100%; margin-top:12px; padding:10px 12px; border:0; border-radius:8px; background:#3b82f6; color:white; cursor:pointer; font-weight:600; }
    button:hover { background:#2563eb; }
    .err { min-height:18px; margin-top:10px; color:#fca5a5; font-size:12px; }
    code { color:#93c5fd; }
  </style>
</head>
<body>
  <form class="card" id="form">
    <h1>__HEADING__</h1>
    <p>__DESCRIPTION__</p>
    <div class="identity-fields">
      <input id="user_id" autocomplete="username" placeholder="用户 ID" autofocus />
      <input id="org_id" placeholder="组织 ID（可选）" />
      <input class="wide" id="password" type="password" autocomplete="current-password" placeholder="用户口令" />
      <input class="wide" id="token" type="password" placeholder="服务访问口令（兼容/可选）" />
    </div>
    <input id="token_fallback" type="password" autocomplete="current-password" placeholder="__PLACEHOLDER__" style="display:__TOKEN_FALLBACK_DISPLAY__" autofocus />
    <div class="login-note">优先使用用户 ID + 用户口令；如果还没创建用户口令，可使用服务访问口令进入默认账号。</div>
    <button type="submit">进入</button>
    <div class="err" id="err"></div>
  </form>
  <script>
    document.getElementById('form').addEventListener('submit', async function (e) {
      e.preventDefault();
      var err = document.getElementById('err');
      err.textContent = '';
      var tokenEl = document.getElementById('token');
      var fallbackEl = document.getElementById('token_fallback');
      var payload = { token: (tokenEl && tokenEl.value) || (fallbackEl && fallbackEl.value) || '' };
      var userEl = document.getElementById('user_id');
      var orgEl = document.getElementById('org_id');
      var passwordEl = document.getElementById('password');
      if (userEl && userEl.value.trim()) payload.user_id = userEl.value.trim();
      if (orgEl && orgEl.value.trim()) payload.org_id = orgEl.value.trim();
      if (passwordEl && passwordEl.value) payload.password = passwordEl.value;
      var r = await fetch('__ENDPOINT__', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if (r.ok) location.href = '__REDIRECT__';
      else err.textContent = '秘钥错误或已过期';
    });
  </script>
</body>
</html>"""
    return (
        template
        .replace("__TITLE__", title)
        .replace("__HEADING__", heading)
        .replace("__DESCRIPTION__", description)
        .replace("__PLACEHOLDER__", input_placeholder)
        .replace("__IDENTITY_DISPLAY__", "grid" if show_identity else "none")
        .replace("__TOKEN_FALLBACK_DISPLAY__", "none" if show_identity else "block")
        .replace("__ENDPOINT__", endpoint)
        .replace("__REDIRECT__", redirect_to)
    )


@app.get("/terminal/api/sessions", dependencies=[Depends(require_terminal_auth)])
async def terminal_list_sessions(request: Request, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    sessions = store.list_sessions(limit=limit, offset=offset, user_id=user_ctx["user"]["id"])
    summaries = _billing_store.session_summaries([s["id"] for s in sessions])
    for session in sessions:
        session["billing"] = summaries.get(session["id"])
        if session.get("user_id"):
            session["quota"] = _quota_store.status(user_id=session["user_id"], org_id=session.get("org_id") or "")
    return sessions


@app.post("/terminal/api/sessions", dependencies=[Depends(require_terminal_auth)])
async def terminal_create_session(request: CreateTerminalSessionRequest, http_request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(http_request)
    if request.session_id:
        _assert_session_access(store.get_session(request.session_id), user_ctx)
    metadata = {**(request.metadata or {}), "user_id": user_ctx["user"]["id"], "org_id": user_ctx["organization"]["id"]}
    if metadata.get("prompt_template"):
        prompt_template = str(metadata["prompt_template"]).strip()
        known_templates = {str(item.get("name") or "") for item in list_prompt_templates()}
        if prompt_template not in known_templates:
            raise HTTPException(status_code=400, detail=f"Prompt template '{prompt_template}' not found")
        metadata["prompt_template"] = prompt_template
    sid = store.create_session(
        session_id=request.session_id,
        metadata=metadata,
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    info = store.get_session(sid)
    return info or {"id": sid, "metadata": request.metadata or {}}


@app.put("/terminal/api/sessions/{session_id}/config", dependencies=[Depends(require_terminal_auth)])
async def terminal_update_session_config(session_id: str, request: TerminalSessionConfigRequest, http_request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(http_request)
    info = store.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    _assert_session_access(info, user_ctx)
    metadata = dict((info or {}).get("metadata") or {})
    if request.prompt_template is not None:
        prompt_template = request.prompt_template.strip() or default_prompt_template_name()
        known_templates = {str(item.get("name") or "") for item in list_prompt_templates()}
        if prompt_template not in known_templates:
            raise HTTPException(status_code=400, detail=f"Prompt template '{prompt_template}' not found")
        metadata["prompt_template"] = prompt_template
    if request.title is not None:
        metadata["title"] = request.title.strip()
    store.update_session_metadata(session_id, metadata)
    return store.get_session(session_id) or {"id": session_id, "metadata": metadata}


@app.get("/terminal/api/prompts", dependencies=[Depends(require_terminal_auth)])
async def terminal_list_prompt_templates() -> dict[str, Any]:
    templates = list_prompt_templates()
    return {"default": default_prompt_template_name(), "templates": templates}


@app.delete("/terminal/api/sessions/{session_id}", dependencies=[Depends(require_terminal_auth)])
async def terminal_delete_session(session_id: str, request: Request) -> dict[str, str]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    ok = store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    _billing_store.delete_session_usage(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/terminal/api/sessions/{session_id}/billing", dependencies=[Depends(require_terminal_auth)])
async def terminal_session_billing(session_id: str, request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    return _billing_store.session_summary(session_id)


@app.get("/terminal/api/sessions/{session_id}/tasks", dependencies=[Depends(require_terminal_auth)])
async def terminal_session_tasks(session_id: str, request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    task_store = get_task_store().for_task_list(session_id)
    tasks = [task.to_dict() for task in task_store.list(include_internal=False)]
    return {"task_list_id": task_store.task_list_id, "tasks_dir": str(task_store.tasks_dir), "tasks": tasks}


@app.post("/terminal/api/sessions/{session_id}/attachments/media", dependencies=[Depends(require_terminal_auth)])
async def terminal_upload_media_attachment(session_id: str, request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    filename = "image"
    content_type = None
    try:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="file field is required")
        data = await upload.read()
        filename = getattr(upload, "filename", None) or "image"
        content_type = getattr(upload, "content_type", None)
    except AssertionError:
        # python-multipart is not installed; fall back to a narrow stdlib parser.
        try:
            filename, content_type, data = _parse_multipart_file(await request.body(), request.headers.get("content-type") or "")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        if "python-multipart" not in str(exc):
            raise
        try:
            filename, content_type, data = _parse_multipart_file(await request.body(), request.headers.get("content-type") or "")
        except ValueError as parse_exc:
            raise HTTPException(status_code=400, detail=str(parse_exc)) from parse_exc
    try:
        item = get_attachment_store().save_media(
            data=data,
            filename=filename,
            content_type=content_type,
            user_id=user_ctx["user"]["id"],
            org_id=user_ctx["organization"]["id"],
            session_id=session_id,
            metadata={"source": "terminal-ui"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"attachment": item.to_dict()}


@app.post("/terminal/api/sessions/{session_id}/attachments/images", dependencies=[Depends(require_terminal_auth)])
async def terminal_upload_image_attachment(session_id: str, request: Request) -> dict[str, Any]:
    return await terminal_upload_media_attachment(session_id, request)


@app.get("/terminal/api/sessions/{session_id}/attachments/{attachment_id}/content", dependencies=[Depends(require_terminal_auth)])
async def terminal_attachment_content(session_id: str, attachment_id: str, request: Request) -> FastAPIResponse:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    item = get_attachment_store().get_authorized(
        attachment_id=attachment_id,
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
        session_id=session_id,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="attachment not found")
    data = get_attachment_store().read_bytes(item)
    disposition = _content_disposition_inline(item.filename)
    range_header = request.headers.get("range") or ""
    if range_header.startswith("bytes="):
        start_text, _, end_text = range_header[6:].partition("-")
        try:
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else len(data) - 1
        except ValueError:
            start, end = 0, len(data) - 1
        start = max(0, min(start, len(data) - 1)) if data else 0
        end = max(start, min(end, len(data) - 1)) if data else 0
        chunk = data[start : end + 1]
        return FastAPIResponse(
            content=chunk,
            status_code=206,
            media_type=item.content_type,
            headers={
                "Cache-Control": "private, max-age=300",
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{len(data)}",
                "Content-Length": str(len(chunk)),
                "Content-Disposition": disposition,
            },
        )
    return FastAPIResponse(
        content=data,
        media_type=item.content_type,
        headers={
            "Cache-Control": "private, max-age=300",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
            "Content-Disposition": disposition,
        },
    )


@app.post("/terminal/api/sessions/{session_id}/image-playground/generate", dependencies=[Depends(require_terminal_auth)])
async def terminal_image_playground_generate(session_id: str, request: ImagePlaygroundRequest, http_request: Request) -> dict[str, Any]:
    return await _terminal_image_playground(session_id=session_id, request=request, http_request=http_request, force_mode="generate")


@app.post("/terminal/api/sessions/{session_id}/image-playground/edit", dependencies=[Depends(require_terminal_auth)])
async def terminal_image_playground_edit(session_id: str, request: ImagePlaygroundRequest, http_request: Request) -> dict[str, Any]:
    return await _terminal_image_playground(session_id=session_id, request=request, http_request=http_request, force_mode="edit")


async def _terminal_image_playground(session_id: str, request: ImagePlaygroundRequest, http_request: Request, *, force_mode: str) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(http_request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    image_client = get_image_generation_client()
    if image_client is None:
        raise HTTPException(status_code=400, detail="Image generation is not configured. Set AGENT_IMAGE_MODEL/API key/base URL.")
    mode = force_mode or request.mode
    attachment_store = get_attachment_store()
    input_images: list[ImageInput] = []
    for attachment_id in request.input_attachment_ids[:16]:
        item = attachment_store.get_authorized(
            attachment_id=attachment_id,
            user_id=user_ctx["user"]["id"],
            org_id=user_ctx["organization"]["id"],
            session_id=session_id,
        )
        if item is None:
            raise HTTPException(status_code=404, detail=f"input attachment not found: {attachment_id}")
        if not item.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"input attachment must be an image: {attachment_id}")
        input_images.append(ImageInput(data=attachment_store.read_bytes(item), content_type=item.content_type, filename=item.filename))
    mask = None
    if request.mask_attachment_id:
        mask_item = attachment_store.get_authorized(
            attachment_id=request.mask_attachment_id,
            user_id=user_ctx["user"]["id"],
            org_id=user_ctx["organization"]["id"],
            session_id=session_id,
        )
        if mask_item is None:
            raise HTTPException(status_code=404, detail=f"mask attachment not found: {request.mask_attachment_id}")
        if not mask_item.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="mask attachment must be an image")
        mask = ImageInput(data=attachment_store.read_bytes(mask_item), content_type=mask_item.content_type, filename=mask_item.filename)
    if mode == "edit" and not input_images:
        raise HTTPException(status_code=400, detail="edit mode requires at least one input image attachment")
    try:
        if mode == "edit":
            result = await image_client.edit_image(
                prompt=request.prompt,
                input_images=input_images,
                mask=mask,
                size=request.size,
                quality=request.quality,
                output_format=request.output_format,
                n=request.n,
            )
        else:
            result = await image_client.generate_image(
                prompt=request.prompt,
                size=request.size,
                quality=request.quality,
                output_format=request.output_format,
                n=request.n,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    attachments = []
    for idx, image in enumerate(result.images, 1):
        item = attachment_store.save_media(
            data=image.data,
            filename=f"image-playground-{mode}-{idx}.{_extension_for_content_type(image.content_type)}",
            content_type=image.content_type,
            user_id=user_ctx["user"]["id"],
            org_id=user_ctx["organization"]["id"],
            session_id=session_id,
            metadata={
                "source": "image-playground",
                "operation": mode,
                "prompt": request.prompt,
                "size": request.size,
                "quality": request.quality,
                "output_format": request.output_format,
                "input_attachment_ids": request.input_attachment_ids,
                "mask_attachment_id": request.mask_attachment_id,
                "revised_prompt": image.revised_prompt,
                "raw_url": image.raw_url,
                "image_metadata": image.metadata,
                "generation_metadata": {k: v for k, v in result.metadata.items() if k != "raw_response"},
            },
        )
        data = item.to_dict()
        data["preview_url"] = f"/terminal/api/sessions/{session_id}/attachments/{item.id}/content"
        attachments.append(data)
    return {
        "attachments": attachments,
        "metadata": {k: v for k, v in result.metadata.items() if k != "raw_response"},
    }


def _extension_for_content_type(content_type: str) -> str:
    if content_type == "image/jpeg":
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    if content_type == "image/gif":
        return "gif"
    return "png"


@app.post("/terminal/api/sessions/{session_id}/tasks/{task_id}/assign", dependencies=[Depends(require_terminal_auth)])
async def terminal_assign_task(session_id: str, task_id: str, request: TaskAssignRequest, http_request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(http_request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    task_store = get_task_store().for_task_list(session_id)
    task = task_store.update(task_id, owner=request.owner)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {"status": "assigned", "task": task.to_dict()}


@app.post("/terminal/api/sessions/{session_id}/tasks/claim-next", dependencies=[Depends(require_terminal_auth)])
async def terminal_claim_next_task(session_id: str, request: Request) -> dict[str, Any]:
    store = create_session_store()
    user_ctx = get_default_user_context(request)
    _assert_session_access(store.get_session(session_id), user_ctx)
    task_store = get_task_store().for_task_list(session_id)
    started = await _maybe_start_watched_task(session_id=session_id, user_id=user_ctx["user"]["id"], org_id=user_ctx["organization"]["id"])
    return {"started": started, "tasks": [task.to_dict() for task in task_store.list(include_internal=False)]}


@app.get("/terminal/api/quota", dependencies=[Depends(require_terminal_auth)])
async def terminal_quota(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    return {
        "user": user_ctx["user"],
        "organization": user_ctx["organization"],
        "quota": _quota_store.status(user_id=user_ctx["user"]["id"], org_id=user_ctx["organization"]["id"]),
        "usage": _billing_store.usage_summary(user_id=user_ctx["user"]["id"]),
    }


@app.get("/terminal/api/config", dependencies=[Depends(require_terminal_auth)])
async def terminal_config(request: Request, session_id: str | None = None) -> dict[str, Any]:
    selection = env_model_selection()
    user_ctx = get_default_user_context(request)
    if session_id:
        _assert_session_access(create_session_store().get_session(session_id), user_ctx)
    prompt_templates = list_prompt_templates()
    return {
        "model_provider": selection["model_provider"],
        "model_id": selection["model_id"],
        "prompt_template": get_session_prompt_template(session_id),
        "default_prompt_template": default_prompt_template_name(),
        "prompt_templates": prompt_templates,
        "codex_credentials": codex_credential_source(),
        "context_window": 200_000,
        "user": user_ctx["user"],
        "organization": user_ctx["organization"],
        "quota": _quota_store.status(
            user_id=user_ctx["user"]["id"],
            org_id=user_ctx["organization"]["id"],
        ),
    }


@app.get("/terminal/api/integrations/codex", dependencies=[Depends(require_terminal_auth)])
async def terminal_codex_status() -> dict[str, Any]:
    return codex_credential_source()


@app.put("/terminal/api/integrations/codex", dependencies=[Depends(require_terminal_auth)])
async def terminal_save_codex_credentials(request: CodexCredentialsRequest) -> dict[str, Any]:
    try:
        credentials = save_codex_credentials(
            secret=request.secret,
            credential_type=request.credential_type,
            base_url=request.base_url,
            model=request.model,
            image_model=request.image_model,
            metadata={**request.metadata, "source": request.metadata.get("source") or "terminal-ui"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    clear_image_generation_client_cache()
    return {"codex": codex_credential_source(), "source": credentials.source, "auth_mode": credentials.auth_mode}


@app.delete("/terminal/api/integrations/codex", dependencies=[Depends(require_terminal_auth)])
async def terminal_delete_codex_credentials() -> dict[str, Any]:
    deleted = delete_codex_credentials()
    clear_image_generation_client_cache()
    return {"deleted": deleted, "codex": codex_credential_source()}


@app.post("/terminal/api/integrations/codex/login", dependencies=[Depends(require_terminal_auth)])
async def terminal_start_codex_login(request: CodexLoginRequest) -> dict[str, Any]:
    try:
        return start_codex_login_flow(flow=request.flow)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/terminal/api/integrations/codex/login", dependencies=[Depends(require_terminal_auth)])
async def terminal_get_codex_login_status() -> dict[str, Any]:
    return codex_login_status()


@app.get("/terminal/api/integrations/feishu", dependencies=[Depends(require_terminal_auth)])
async def terminal_feishu_status(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    status_payload = _feishu_token_store.status(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    ).to_dict()
    status_payload["bridge"] = _feishu_ws_bridge.status(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    return status_payload


@app.put("/terminal/api/integrations/feishu/token", dependencies=[Depends(require_terminal_auth)])
async def terminal_save_feishu_token(request: FeishuTokenRequest, http_request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(http_request)
    token = _feishu_token_store.save_user_token(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
        access_token=request.access_token,
        refresh_token=request.refresh_token,
        expires_in=request.expires_in,
        expires_at=request.expires_at,
        scopes=request.scopes,
        token_type=request.token_type,
        metadata=request.metadata,
    )
    return {"status": "connected", "feishu": token}


@app.put("/terminal/api/integrations/feishu/app-credentials", dependencies=[Depends(require_terminal_auth)])
async def terminal_save_feishu_app_credentials(request: FeishuAppCredentialsRequest, http_request: Request) -> dict[str, Any]:
    parsed = _parse_feishu_app_credential(request.credential)
    user_ctx = get_default_user_context(http_request)
    existing = _feishu_token_store.get_user_token(user_id=user_ctx["user"]["id"], org_id=user_ctx["organization"]["id"])
    existing_metadata = dict((existing or {}).get("metadata") or {})
    token = _feishu_token_store.save_app_credentials(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
        app_id=parsed["app_id"],
        app_secret=parsed["app_secret"],
        metadata={**existing_metadata, "source": "terminal-ui", "mode": "app_credentials", **request.metadata},
    )
    return {"status": "connected", "feishu": token}


@app.post("/terminal/api/integrations/feishu/test", dependencies=[Depends(require_terminal_auth)])
async def terminal_test_feishu_connection(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    user_id = user_ctx["user"]["id"]
    org_id = user_ctx["organization"]["id"]
    status = _feishu_token_store.status(user_id=user_id, org_id=org_id).to_dict()
    test_path = "/open-apis/bot/v3/info" if status.get("credential_type") == "app_credentials" else "/open-apis/authen/v1/user_info"
    result = await FeishuApiTool(_feishu_token_store).call(
        {"method": "GET", "path": test_path},
        ToolContext(
            session_id="terminal-feishu-test",
            messages=[],
            metadata={"user_id": user_id, "org_id": org_id},
        ),
    )
    try:
        payload: Any = json.loads(result.content)
    except json.JSONDecodeError:
        payload = {"message": result.content}

    status_code = int(payload.get("status") or 200) if isinstance(payload, dict) else 200
    feishu_code = payload.get("code") if isinstance(payload, dict) else None
    ok = not result.is_error and status_code < 400 and feishu_code in (None, 0)
    message = "飞书连接测试成功" if ok else "飞书连接测试失败"
    if isinstance(payload, dict) and payload.get("msg"):
        message = f"{message}: {payload['msg']}"
    elif result.is_error:
        message = str(payload.get("message") if isinstance(payload, dict) else result.content)

    return {
        "ok": ok,
        "message": message,
        "user_id": user_id,
        "org_id": org_id,
        "path": test_path,
        "result": payload,
    }


@app.get("/terminal/api/integrations/feishu/bridge", dependencies=[Depends(require_terminal_auth)])
async def terminal_feishu_bridge_status(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    return _feishu_ws_bridge.status(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )


@app.get("/terminal/api/integrations/feishu/bridge/logs", dependencies=[Depends(require_terminal_auth)])
async def terminal_feishu_bridge_logs(request: Request, limit: int = 100) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    return _feishu_ws_bridge.logs(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
        limit=limit,
    )


@app.post("/terminal/api/integrations/feishu/bridge/start", dependencies=[Depends(require_terminal_auth)])
async def terminal_start_feishu_bridge(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    user_id = user_ctx["user"]["id"]
    org_id = user_ctx["organization"]["id"]
    try:
        bridge = _feishu_ws_bridge.start(
            user_id=user_id,
            org_id=org_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _feishu_token_store.update_metadata(user_id=user_id, org_id=org_id, metadata={"bridge_autostart": True})
    return {"status": bridge.get("state") or "starting", "bridge": bridge}


@app.post("/terminal/api/integrations/feishu/bridge/stop", dependencies=[Depends(require_terminal_auth)])
async def terminal_stop_feishu_bridge(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    user_id = user_ctx["user"]["id"]
    org_id = user_ctx["organization"]["id"]
    bridge = _feishu_ws_bridge.stop(
        user_id=user_id,
        org_id=org_id,
    )
    _feishu_token_store.update_metadata(user_id=user_id, org_id=org_id, metadata={"bridge_autostart": False})
    return {"status": bridge.get("state") or "stopped", "bridge": bridge}


@app.delete("/terminal/api/integrations/feishu", dependencies=[Depends(require_terminal_auth)])
async def terminal_delete_feishu_token(request: Request) -> dict[str, Any]:
    user_ctx = get_default_user_context(request)
    _feishu_ws_bridge.stop(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    deleted = _feishu_token_store.delete_user_token(
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    return {"status": "disconnected", "deleted": deleted}


@app.post("/agent/run", dependencies=[Depends(require_terminal_auth)])
async def run_agent(request: AgentRequest, http_request: Request) -> dict[str, Any]:
    """非流式调试：收集完整事件数组后返回。"""
    user_ctx = get_default_user_context(http_request)
    if request.session_id:
        _assert_session_access(create_session_store().get_session(request.session_id), user_ctx)
    runtime = create_runtime(
        mode=request.mode,
        session_id=request.session_id,
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    events: list[AgentEvent] = []
    async for event in runtime.run(request.message):
        events.append(event)
    return {"events": [{"type": event.type, "data": event.data} for event in events]}


@app.post("/agent/stream", dependencies=[Depends(require_terminal_auth)])
async def stream_agent(request: AgentRequest, http_request: Request) -> StreamingResponse:
    """流式调试：SSE 事件流。"""
    user_ctx = get_default_user_context(http_request)
    if request.session_id:
        _assert_session_access(create_session_store().get_session(request.session_id), user_ctx)
    runtime = create_runtime(
        mode=request.mode,
        session_id=request.session_id,
        user_id=user_ctx["user"]["id"],
        org_id=user_ctx["organization"]["id"],
    )
    return StreamingResponse(
        agent_sse(runtime, request.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── WebSocket: PTY terminal ──────────────────────────────────


@app.websocket("/terminal/pty")
async def terminal_pty(websocket: WebSocket, mode: str = "fake_tool") -> None:
    """xterm.js <-> WebSocket <-> PTY <-> Agent CLI。"""
    if not is_terminal_ws_authenticated(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    master_fd, slave_fd = os.openpty()
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-m", "agent_core.examples.repl", "--mode", mode,
         "--session-id", f"web-pty-{id(websocket)}"],
        stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=env, start_new_session=True, close_fds=True,
    )
    os.close(slave_fd)

    async def read_pty() -> None:
        try:
            while True:
                chunk = await asyncio.to_thread(os.read, master_fd, 4096)
                if not chunk:
                    break
                await websocket.send_text(chunk.decode("utf-8", errors="replace"))
        except Exception:
            with contextlib.suppress(Exception):
                await websocket.close()

    async def write_pty() -> None:
        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    os.write(master_fd, message["text"].encode("utf-8"))
                elif "bytes" in message and message["bytes"] is not None:
                    os.write(master_fd, message["bytes"])
                elif message.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            return

    try:
        done, pending = await asyncio.wait(
            {asyncio.create_task(read_pty()), asyncio.create_task(write_pty())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(Exception):
            process.wait(timeout=2)
        with contextlib.suppress(OSError):
            os.close(master_fd)


# ── WebSocket: Direct Agent (Channel-based) ──────────────────


def _format_task_prompt(task: Any) -> str:
    prompt = f"Complete all open tasks. Start with task #{task.id}:\n\n{task.subject}"
    if task.description:
        prompt += f"\n\n{task.description}"
    prompt += "\n\nUse TaskGet to read the latest task state, TaskUpdate to mark it in_progress before working, and TaskUpdate status=completed only after the task is fully done."
    return prompt


def _task_workspace_slug(session_id: str, task_id: str) -> str:
    safe_session = re.sub(r"[^a-zA-Z0-9._-]+", "-", session_id).strip("-._") or "session"
    return f"task-{safe_session[:40]}-{task_id}"[:64]


def _permission_request_matches_pending(event_data: dict[str, Any], pending: dict[str, Any] | None) -> bool:
    if not pending:
        return False
    event_id = str(event_data.get("tool_use_id") or "")
    pending_id = str(pending.get("tool_use_id") or "")
    if event_id or pending_id:
        return bool(event_id and event_id == pending_id)
    return event_data.get("tool") == pending.get("tool") and event_data.get("input") == pending.get("input")


async def _maybe_start_watched_task(*, session_id: str, user_id: str, org_id: str) -> bool:
    if await agent_task_manager.is_running(session_id):
        return False
    task_store = get_task_store().for_task_list(session_id)
    available = task_store.available_tasks()
    if not available:
        return False
    task = available[0]
    workspace_info = None
    workspace_cwd: str | None = None
    workspace_status = "unavailable"
    worktree_manager = create_worktree_manager()
    if await worktree_manager.can_create():
        try:
            workspace_info = await worktree_manager.create_agent_worktree(
                _task_workspace_slug(session_id, task.id),
                session_id=session_id,
            )
            workspace_cwd = workspace_info.worktree_path
            workspace_status = "active"
        except Exception as exc:
            task_store.update(task.id, metadata={"workspace": {"status": "failed", "error": str(exc)}})
            workspace_status = "failed"
    claim = task_store.claim(task.id, session_id, check_agent_busy=True)
    if not claim.success or not claim.task:
        if workspace_info:
            with contextlib.suppress(Exception):
                await worktree_manager.remove_agent_worktree(workspace_info)
        return False
    if workspace_info:
        claim.task = task_store.update(claim.task.id, metadata={"workspace": {**workspace_info.to_dict(), "status": workspace_status, "label": "isolated"}}) or claim.task
    runtime = create_runtime(session_id=session_id, ask_callback=None, user_id=user_id, org_id=org_id, cwd=workspace_cwd)
    runtime._config.metadata["task_id"] = claim.task.id
    if workspace_info:
        runtime._config.metadata["workspace"] = {**workspace_info.to_dict(), "status": workspace_status, "label": "isolated"}
    managed = await agent_task_manager.start(
        session_id=session_id,
        runtime=runtime,
        message=_format_task_prompt(claim.task),
        workspace_manager=worktree_manager if workspace_info else None,
        workspace=workspace_info,
        task_store=task_store,
        task_id=claim.task.id,
    )
    return managed is not None


@app.websocket("/terminal/ws")
async def terminal_ws(websocket: WebSocket, session_id: str | None = None) -> None:
    """聊天 UI 双向通道 — 通过 Channel 抽象与 AgentRuntime 交互。

    前端发送完整文本消息（非逐按键），后端返回 JSON 事件流。
    权限交互：后端发送 JSON 卡片 → 前端按钮 → JSON 响应
    """
    if not is_terminal_ws_authenticated(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    effective_session_id = session_id or f"ws-{id(websocket)}"
    command_session_store = create_session_store()
    user_ctx = get_default_user_context(websocket=websocket)
    effective_user_id = user_ctx["user"]["id"]
    effective_org_id = user_ctx["organization"]["id"]
    if not _assert_ws_session_access(command_session_store.get_session(effective_session_id), user_ctx):
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Session does not belong to current user",
        }, ensure_ascii=False))
        await websocket.close(code=1008)
        return

    await websocket.send_text(json.dumps({
        "type": "slash_commands",
        "commands": command_specs(),
    }, ensure_ascii=False))
    await websocket.send_text(json.dumps({
        "type": "companion_state",
        "companion": companion_payload(get_companion(user_ctx, create=False)),
    }, ensure_ascii=False, default=str))

    runtime = create_runtime(session_id=effective_session_id, ask_callback=None, user_id=effective_user_id, org_id=effective_org_id)
    plan_meta = export_plan_metadata(effective_session_id, cwd=runtime._config.cwd)
    if plan_meta.get("plan_slug") or plan_meta.get("mode") == "plan":
        await websocket.send_text(json.dumps({
            "type": "plan_state",
            **plan_meta,
            "plan": get_plan(effective_session_id, cwd=runtime._config.cwd),
        }, ensure_ascii=False, default=str))

    async def compact_session(target_session_id: str, custom_instructions: str | None = None) -> dict[str, Any]:
        compact_runtime = create_runtime(session_id=target_session_id, ask_callback=None, user_id=effective_user_id, org_id=effective_org_id)
        result = await compact_runtime.compact_history(custom_instructions=custom_instructions)
        return {
            "pre_message_count": result.pre_message_count,
            "post_message_count": result.post_message_count,
            "pre_token_count": result.pre_token_count,
            "post_token_count": result.post_token_count,
            "summary": result.summary,
        }

    # ── 回放历史消息（JSON 格式）──
    from agent_core.types import TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock

    tool_names_by_id: dict[str, str] = {}
    for msg in runtime.messages:
        for block in msg.content:
            if isinstance(block, TextBlock):
                if msg.metadata.get("compact_summary"):
                    await websocket.send_text(json.dumps({
                        "type": "history_compact_summary",
                        "text": msg.metadata.get("summary") or block.text,
                        "pre_message_count": msg.metadata.get("pre_compact_message_count"),
                    }, ensure_ascii=False))
                elif msg.role == "user":
                    await websocket.send_text(json.dumps({
                        "type": "history_user", "text": msg.metadata.get("original_user_input") or block.text,
                        "attachments": msg.metadata.get("attachments") or [],
                    }, ensure_ascii=False))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "history_assistant", "text": block.text,
                    }, ensure_ascii=False))
            elif isinstance(block, ToolUseBlock):
                tool_names_by_id[block.id] = block.name
                await websocket.send_text(json.dumps({
                    "type": "history_tool_use", "tool": block.name, "input": block.input,
                    "tool_use_id": block.id,
                }, ensure_ascii=False))
            elif isinstance(block, ToolResultBlock):
                tool_name = tool_names_by_id.get(block.tool_use_id, "?")
                await websocket.send_text(json.dumps({
                    "type": "history_tool_result", "tool": tool_name,
                    "tool_use_id": block.tool_use_id,
                    "result": block.content[:500], "is_error": block.is_error,
                    "metadata": block.metadata,
                }, ensure_ascii=False))
            elif isinstance(block, ThinkingBlock):
                await websocket.send_text(json.dumps({
                    "type": "history_thinking", "text": block.thinking,
                }, ensure_ascii=False))

    last_task_seq = 0

    async def send_task_state() -> None:
        task_store = get_task_store().for_task_list(effective_session_id)
        await websocket.send_text(json.dumps({
            "type": "task_state",
            "task_list_id": task_store.task_list_id,
            "tasks": [task.to_dict() for task in task_store.list(include_internal=False)],
        }, ensure_ascii=False, default=str))

    await send_task_state()

    async def send_pending_permission_request() -> None:
        """Re-emit an outstanding permission prompt for reconnect/race recovery."""
        pending = await agent_task_manager.pending_permission(effective_session_id)
        if not pending:
            return
        await websocket.send_text(json.dumps({
            "type": "permission_request",
            **pending,
            "replayed": True,
        }, ensure_ascii=False, default=str))

    async def drain_task_events() -> None:
        """Send any buffered background task events newer than last_task_seq."""
        nonlocal last_task_seq
        events, _, _ = await agent_task_manager.events_after(effective_session_id, last_task_seq)
        saw_task_change = False
        saw_terminal = False
        for seq, event in events:
            last_task_seq = max(last_task_seq, seq)
            if event is None:
                continue
            if event.type in ("loop_completed", "loop_failed", "loop_aborted"):
                last_task_seq = seq
                saw_terminal = True
            if event.type == "tool_completed" and str(event.data.get("tool") or "").startswith("Task"):
                saw_task_change = True
            if event.type == "task_state":
                saw_task_change = False
            if event.type == "workspace_state":
                saw_task_change = True
            if event.type == "permission_request":
                pending_permission = await agent_task_manager.pending_permission(effective_session_id)
                if not _permission_request_matches_pending(event.data, pending_permission):
                    continue
            chat_json = event_to_chat_json(event)
            if chat_json:
                await websocket.send_text(chat_json)
        if saw_task_change:
            await send_task_state()
        if saw_terminal:
            if await _maybe_start_watched_task(session_id=effective_session_id, user_id=effective_user_id, org_id=effective_org_id):
                last_task_seq = 0
                await websocket.send_text(json.dumps({
                    "type": "agent_status",
                    "running": True,
                    "session_id": effective_session_id,
                    "reason": "task_watcher",
                }, ensure_ascii=False))

    running_task = await agent_task_manager.get(effective_session_id)
    if running_task and not running_task.done:
        await websocket.send_text(json.dumps({
            "type": "agent_status",
            "running": True,
            "session_id": effective_session_id,
        }, ensure_ascii=False))
        await drain_task_events()
        await send_pending_permission_request()

    try:
        while True:
            recv_msg = asyncio.create_task(websocket.receive_text())
            wait_task: asyncio.Task | None = None
            running_task = await agent_task_manager.get(effective_session_id)
            if running_task and not running_task.done:
                wait_task = asyncio.create_task(agent_task_manager.wait_for_next(effective_session_id, last_task_seq))

            wait_set = {recv_msg}
            if wait_task:
                wait_set.add(wait_task)
            done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
            for pending_task in pending:
                pending_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_task

            if wait_task and wait_task in done:
                await drain_task_events()
                await send_pending_permission_request()
                continue

            if recv_msg not in done:
                continue

            data = recv_msg.result()

            # 尝试解析 JSON（权限响应、中断等结构化消息）
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    if parsed.get("type") == "permission_response":
                        await agent_task_manager.respond_permission(
                            effective_session_id,
                            "allow" if parsed.get("decision") == "allow" else "deny",
                            parsed.get("option") if isinstance(parsed.get("option"), dict) else None,
                        )
                    elif parsed.get("type") == "abort":
                        await agent_task_manager.abort(effective_session_id)
                    elif parsed.get("type") == "user_message":
                        line = str(parsed.get("text") or "").strip()
                        raw_attachments = parsed.get("attachments") if isinstance(parsed.get("attachments"), list) else []
                        attachments: list[dict[str, Any]] = []
                        for raw in raw_attachments:
                            if not isinstance(raw, dict):
                                continue
                            attachment_id = str(raw.get("id") or raw.get("attachment_id") or "").strip()
                            if not attachment_id:
                                continue
                            item = get_attachment_store().get_authorized(
                                attachment_id=attachment_id,
                                user_id=effective_user_id,
                                org_id=effective_org_id,
                                session_id=effective_session_id,
                            )
                            if item is None:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": f"Attachment not found or not accessible: {attachment_id}",
                                }, ensure_ascii=False))
                                continue
                            attachments.append(item.to_dict())
                        if not line and not attachments:
                            continue
                        if line.startswith("/") and not attachments:
                            result = await handle_slash_command(
                                line,
                                session_id=effective_session_id,
                                session_store=command_session_store,
                                is_running=agent_task_manager.is_running,
                                abort=agent_task_manager.abort,
                                compact=compact_session,
                                user_context=user_ctx,
                            )
                            await websocket.send_text(json.dumps(result, ensure_ascii=False, default=str))
                            continue
                        if await agent_task_manager.is_running(effective_session_id):
                            await websocket.send_text(json.dumps({
                                "type": "agent_status",
                                "running": True,
                                "session_id": effective_session_id,
                            }, ensure_ascii=False))
                            continue
                        turn_runtime = create_runtime(session_id=effective_session_id, ask_callback=None, user_id=effective_user_id, org_id=effective_org_id)
                        managed = await agent_task_manager.start(
                            session_id=effective_session_id,
                            runtime=turn_runtime,
                            message=line or "Please inspect the attached media file(s).",
                            attachments=attachments,
                        )
                        if managed is None:
                            continue
                        last_task_seq = 0
                        await websocket.send_text(json.dumps({
                            "type": "agent_status",
                            "running": True,
                            "session_id": effective_session_id,
                        }, ensure_ascii=False))
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # 纯文本消息 → 提交给 session 级后台 agent task
            line = data.strip()
            if not line:
                continue
            if line.startswith("/"):
                result = await handle_slash_command(
                    line,
                    session_id=effective_session_id,
                    session_store=command_session_store,
                    is_running=agent_task_manager.is_running,
                    abort=agent_task_manager.abort,
                    compact=compact_session,
                    user_context=user_ctx,
                )
                await websocket.send_text(json.dumps(result, ensure_ascii=False, default=str))
                if "companion" in result:
                    await websocket.send_text(json.dumps({
                        "type": "companion_state",
                        "companion": result.get("companion"),
                        "buddy_action": result.get("buddy_action"),
                    }, ensure_ascii=False, default=str))
                if result.get("clear_terminal"):
                    last_task_seq = 0
                continue
            if await agent_task_manager.is_running(effective_session_id):
                await websocket.send_text(json.dumps({
                    "type": "agent_status",
                    "running": True,
                    "session_id": effective_session_id,
                }, ensure_ascii=False))
                continue

            # Re-create the runtime at turn start instead of reusing the one
            # created when this WebSocket connected.  A reconnect may happen
            # while a previous background task is still running; that stale
            # runtime would not include messages persisted after the task
            # completes and could overwrite them on the next persist.
            turn_runtime = create_runtime(session_id=effective_session_id, ask_callback=None, user_id=effective_user_id, org_id=effective_org_id)
            managed = await agent_task_manager.start(
                session_id=effective_session_id,
                runtime=turn_runtime,
                message=line,
            )
            if managed is None:
                continue
            last_task_seq = 0
            await websocket.send_text(json.dumps({
                "type": "agent_status",
                "running": True,
                "session_id": effective_session_id,
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        return
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.close()
