"""Admin API — Prompt 模板管理、运行时配置查看、会话 CRUD。"""
from __future__ import annotations

import os
import json
import re
import secrets
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from agent_core.billing.store import BillingStore
from agent_core.context.compact import AutoCompactConfig
from agent_core.core.agent import AgentRuntimeConfig
from agent_core.memory.session_memory import SessionMemoryConfig, SessionMemoryManager
from agent_core.permissions.policy import USER_PERMISSION_MODE_CHOICES, get_session_permission_state, permission_mode_title
from agent_core.quota.store import QuotaStore
from agent_core.sandbox import get_sandbox_manager
from agent_core.skills.store import SkillStore
from agent_core.session.store import create_session_store, serialize_message, deserialize_message
from agent_core.users.store import UserStore
from agent_core.hooks.store import HookStore
from agent_core.hooks.types import HookEvent
from agent_server.codex_credentials import codex_credential_source, codex_login_status, delete_codex_credentials, save_codex_credentials, start_codex_login_flow
from agent_server.model_providers import configured_provider_names, env_model_selection
from agent_server.prompt_store import PromptStore, SOUL_DESIGN_PRINCIPLES, clear_system_prompt_cache
from agent_server.runtime_factory import clear_image_generation_client_cache

_ADMIN_COOKIE_NAME = "my_agent_admin_token"
_GENERATED_ADMIN_TOKEN = secrets.token_urlsafe(32)
_GENERATED_TOKEN_PRINTED = False


def get_admin_token() -> str:
    """Return the configured Admin token, or a per-process fallback token.

    推荐生产/长期使用时显式设置 AGENT_ADMIN_TOKEN。未设置时生成进程级
    临时 token，避免 Admin API 默认裸奔。
    """
    return (
        os.getenv("AGENT_ADMIN_TOKEN")
        or os.getenv("ADMIN_TOKEN")
        or os.getenv("AGENT_ADMIN_KEY")
        or _GENERATED_ADMIN_TOKEN
    )


def print_generated_admin_token_once() -> None:
    global _GENERATED_TOKEN_PRINTED
    if _GENERATED_TOKEN_PRINTED:
        return
    if os.getenv("AGENT_ADMIN_TOKEN") or os.getenv("ADMIN_TOKEN") or os.getenv("AGENT_ADMIN_KEY"):
        return
    _GENERATED_TOKEN_PRINTED = True
    print(
        "[admin] AGENT_ADMIN_TOKEN is not set. Generated temporary admin token for this process:\n"
        f"[admin] {_GENERATED_ADMIN_TOKEN}\n"
        "[admin] Set AGENT_ADMIN_TOKEN to a stable secret for persistent access.",
        file=sys.stderr,
    )


def _extract_admin_token(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("x-admin-token")
        or request.cookies.get(_ADMIN_COOKIE_NAME)
        or request.query_params.get("admin_token")
        or ""
    )


def is_admin_authenticated(request: Request) -> bool:
    supplied = _extract_admin_token(request)
    expected = get_admin_token()
    return bool(supplied) and secrets.compare_digest(supplied, expected)


async def require_admin_auth(request: Request) -> None:
    if not is_admin_authenticated(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(prefix="/admin/api", tags=["admin"], dependencies=[Depends(require_admin_auth)])

# 模块级单例，与 runtime_factory 共享
_prompt_store: PromptStore | None = None
_hook_store: HookStore | None = None
_skill_store: SkillStore | None = None
_billing_store: BillingStore | None = None
_user_store: UserStore | None = None
_quota_store: QuotaStore | None = None


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_prompt_store() -> PromptStore:
    global _prompt_store
    if _prompt_store is None:
        _prompt_store = PromptStore()
    return _prompt_store


def get_session_store():
    return create_session_store()


def get_billing_store() -> BillingStore:
    global _billing_store
    if _billing_store is None:
        _billing_store = BillingStore()
    return _billing_store


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


def get_quota_store() -> QuotaStore:
    global _quota_store
    if _quota_store is None:
        _quota_store = QuotaStore()
    return _quota_store


def get_hook_store() -> HookStore:
    global _hook_store
    if _hook_store is None:
        from pathlib import Path
        _hook_store = HookStore(hooks_dir=Path(__file__).resolve().parent / "hooks")
    return _hook_store


def get_skill_store() -> SkillStore:
    global _skill_store
    if _skill_store is None:
        from pathlib import Path
        _skill_store = SkillStore(cwd=Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve())
    return _skill_store


# ── Pydantic Models ──────────────────────────────────────────


class TemplateMetadata(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"


class TemplateUpsertRequest(BaseModel):
    """创建/更新模板请求体。"""
    metadata: TemplateMetadata
    product_name: str = "MyAgent"
    base_instructions: str = ""
    design_principles: list[dict[str, Any]] = []
    sections: list[dict[str, str]] = []
    dynamic_sections: list[dict[str, Any]] = []
    context: dict[str, Any] = {}


class CreateSessionRequest(BaseModel):
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class UpdateSessionRequest(BaseModel):
    metadata: dict[str, Any] | None = None


class ModelPricingRequest(BaseModel):
    model_id: str
    display_name: str = ""
    provider: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cache_read_per_million: float = 0.0
    cache_write_per_million: float = 0.0
    currency: str = "CNY"


class UserUpsertRequest(BaseModel):
    id: str | None = None
    account_uuid: str | None = None
    email: str = ""
    name: str = ""
    status: str = "active"
    subscription_type: str | None = None
    rate_limit_tier: str | None = None
    password: str | None = None
    metadata: dict[str, Any] = {}


class OrganizationUpsertRequest(BaseModel):
    id: str | None = None
    organization_uuid: str | None = None
    name: str = ""
    billing_type: str | None = None
    status: str = "active"
    metadata: dict[str, Any] = {}


class OrganizationMemberRequest(BaseModel):
    user_id: str
    org_id: str
    role: str = "member"


class QuotaPolicyRequest(BaseModel):
    id: str | None = None
    scope_type: str = "default"
    scope_id: str = ""
    name: str = ""
    max_requests_per_day: int = 0
    max_tokens_per_day: int = 0
    max_cost_per_day: float = 0.0
    max_requests_per_week: int = 0
    max_tokens_per_week: int = 0
    max_cost_per_week: float = 0.0
    max_requests_per_month: int = 0
    max_tokens_per_month: int = 0
    max_cost_per_month: float = 0.0
    enabled: bool = True
    metadata: dict[str, Any] = {}


class CodexCredentialsRequest(BaseModel):
    secret: str
    credential_type: str = "api_key"
    base_url: str = "http://localhost:8080/v1"
    model: str = "gpt-5.4"
    image_model: str = "gpt-image-2"
    metadata: dict[str, Any] = {}


class CodexLoginRequest(BaseModel):
    flow: str = "browser"


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    content: str | None = None


def _skill_to_dict(skill: Any) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "when_to_use": skill.when_to_use,
        "source": skill.source,
        "path": str(skill.path),
        "base_dir": str(skill.base_dir),
        "aliases": skill.aliases,
        "argument_hint": skill.argument_hint,
        "allowed_tools": skill.allowed_tools,
        "version": skill.version,
        "model": skill.model,
        "context": skill.execution_context,
        "agent": skill.agent,
        "effort": skill.effort,
        "shell": skill.shell,
        "paths": skill.paths,
        "argument_names": skill.argument_names,
        "disable_model_invocation": skill.disable_model_invocation,
        "user_invocable": skill.user_invocable,
        "editable": _is_project_skill_path(Path(skill.path)),
    }


_SAFE_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _workspace_root() -> Path:
    return Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()


def _project_skill_root() -> Path:
    return _workspace_root() / ".claude" / "skills"


def _validate_skill_name(name: str) -> str:
    cleaned = name[1:] if name.startswith("/") else name
    cleaned = cleaned.strip()
    if not cleaned or not _SAFE_SKILL_NAME_RE.fullmatch(cleaned):
        raise ValueError("Skill name must match [A-Za-z0-9_.:-]+")
    if cleaned in {".", ".."} or ".." in cleaned.split(":"):
        raise ValueError("Skill name cannot contain path traversal")
    return cleaned


def _project_skill_file(name: str) -> Path:
    safe = _validate_skill_name(name)
    root = _project_skill_root()
    path = (root / safe / "SKILL.md").resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Skill path escapes project skill root")
    return path


def _is_project_skill_path(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
        root = _project_skill_root().resolve()
        return resolved == root or root in resolved.parents
    except Exception:
        return False


def _require_editable_skill_file(store: SkillStore, name: str) -> Path:
    skill = store.get_skill(name)
    if skill is None:
        raise FileNotFoundError(f"Skill '{name}' not found")
    path = Path(skill.path).expanduser().resolve()
    if not _is_project_skill_path(path):
        raise PermissionError("Only project skills under .claude/skills are editable from Web Admin")
    return path


def _default_skill_markdown(name: str, description: str = "") -> str:
    import json
    desc = description or f"Skill {name}"
    yaml_name = json.dumps(name, ensure_ascii=False)
    yaml_desc = json.dumps(desc, ensure_ascii=False)
    return f"""---
name: {yaml_name}
description: {yaml_desc}
---
# {name}

Describe when to use this skill and the workflow the agent should follow.
"""


def _clear_skill_runtime_caches(store: SkillStore) -> None:
    store.clear_cache()
    clear_system_prompt_cache()
    try:
        from agent_server import runtime_factory
        runtime_factory._skill_store.clear_cache()
    except Exception:
        pass


def _model_pricing_to_text(model: dict[str, Any]) -> str:
    return json.dumps(model, ensure_ascii=False, indent=2)


def _model_pricing_preview(model: dict[str, Any]) -> str:
    currency = model.get("currency") or "CNY"
    return (
        f"Model: {model.get('display_name') or model.get('model_id')}\n"
        f"ID: {model.get('model_id')}\n"
        f"Provider: {model.get('provider') or '-'}\n"
        f"Pricing: {currency} {model.get('input_per_million', 0)}/M input · "
        f"{currency} {model.get('output_per_million', 0)}/M output\n"
        f"Cache: {currency} {model.get('cache_read_per_million', 0)}/M read hit · "
        f"{currency} {model.get('cache_write_per_million', 0)}/M write\n"
        f"Updated: {model.get('updated_at') or '-'}"
    )


# ── Prompt CRUD ──────────────────────────────────────────────


@router.get("/prompts")
async def list_prompts() -> list[dict[str, Any]]:
    """列出所有 prompt 模板。"""
    store = get_prompt_store()
    return store.list_templates()


@router.get("/prompts/{name}")
async def get_prompt(name: str, raw: bool = False) -> Any:
    """获取指定 prompt 模板。raw=True 时返回原始 YAML 文本。"""
    store = get_prompt_store()
    try:
        if raw:
            from pathlib import Path
            path = store._resolve_path(name)
            if not path.exists():
                raise FileNotFoundError(f"Template '{name}' not found")
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/yaml")
        return store.get_template(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")


@router.get("/prompt-design-principles/default")
async def default_prompt_design_principles() -> dict[str, Any]:
    """返回内置 SOUL 风格设计原则，供 Admin UI 一键写入模板。"""
    return {"source": "openclaw/docs/reference/templates/SOUL.md", "principles": SOUL_DESIGN_PRINCIPLES}


@router.post("/prompts")
async def create_prompt(request: TemplateUpsertRequest) -> dict[str, Any]:
    """创建新 prompt 模板。"""
    store = get_prompt_store()
    try:
        data = request.model_dump()
        return store.create_template(data)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/prompts/{name}")
async def update_prompt(name: str, request: TemplateUpsertRequest) -> dict[str, Any]:
    """更新 prompt 模板。"""
    store = get_prompt_store()
    try:
        data = request.model_dump()
        return store.update_template(name, data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/prompts/{name}/raw")
async def update_prompt_raw(name: str, request: Request) -> dict[str, Any]:
    """通过原始 YAML 文本更新 prompt 模板。"""
    store = get_prompt_store()
    import yaml
    try:
        body = await request.body()
        text = body.decode("utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Template must be a YAML mapping")
        return store.update_template(name, data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/prompts/{name}")
async def delete_prompt(name: str) -> dict[str, str]:
    """删除 prompt 模板。"""
    store = get_prompt_store()
    try:
        store.delete_template(name)
        return {"status": "deleted", "name": name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")


@router.get("/prompts/{name}/preview")
async def preview_prompt(name: str) -> dict[str, str]:
    """预览渲染后的 system prompt。"""
    store = get_prompt_store()
    try:
        rendered = await store.preview(name)
        return {"name": name, "rendered": rendered}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")


# ── Skill Discovery ──────────────────────────────────────────


@router.get("/skills")
async def list_skills(refresh: bool = False) -> list[dict[str, Any]]:
    """列出本地 skills。"""
    store = get_skill_store()
    if refresh:
        _clear_skill_runtime_caches(store)
    return [_skill_to_dict(s) for s in store.list_all_skills(refresh=refresh)]


@router.post("/skills")
async def create_skill(request: SkillCreateRequest) -> dict[str, Any]:
    """在项目 .claude/skills/<name>/SKILL.md 创建新 skill。"""
    store = get_skill_store()
    try:
        name = _validate_skill_name(request.name)
        path = _project_skill_file(name)
        if path.exists():
            raise FileExistsError(f"Project skill '{name}' already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = request.content if request.content is not None else _default_skill_markdown(name, request.description)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        _clear_skill_runtime_caches(store)
        skill = store.get_skill(name)
        if skill is None:
            raise ValueError("Skill file was written but could not be loaded")
        data = _skill_to_dict(skill)
        data["content"] = skill.content
        return data
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/skills/{name}")
async def get_skill(name: str, raw: bool = False) -> Any:
    """获取指定 skill 元信息与内容。raw=True 时返回完整 SKILL.md。"""
    store = get_skill_store()
    skill = store.get_skill(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    if raw:
        return PlainTextResponse(Path(skill.path).read_text(encoding="utf-8"), media_type="text/markdown")
    data = _skill_to_dict(skill)
    data["content"] = skill.content
    return data


@router.put("/skills/{name}/raw")
async def update_skill_raw(name: str, request: Request) -> dict[str, Any]:
    """通过完整 SKILL.md 文本更新项目 skill。"""
    store = get_skill_store()
    try:
        path = _require_editable_skill_file(store, name)
        body = await request.body()
        text = body.decode("utf-8")
        if not text.strip():
            raise ValueError("Skill content cannot be empty")
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        _clear_skill_runtime_caches(store)
        updated = store.get_skill(name)
        if updated is None:
            raise ValueError("Updated skill could not be loaded")
        data = _skill_to_dict(updated)
        data["content"] = updated.content
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/skills/{name}")
async def delete_skill(name: str) -> dict[str, str]:
    """删除项目 skill。只允许删除 .claude/skills 下的项目 skill。"""
    store = get_skill_store()
    try:
        path = _require_editable_skill_file(store, name)
        skill_dir = path.parent
        path.unlink()
        try:
            skill_dir.rmdir()
        except OSError:
            pass
        _clear_skill_runtime_caches(store)
        return {"status": "deleted", "name": name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.get("/skills/{name}/render")
async def render_skill(name: str, args: str = "", session_id: str = "") -> dict[str, str]:
    """渲染 Skill tool 实际注入模型的完整内容。"""
    store = get_skill_store()
    try:
        return {"name": name, "rendered": store.render_skill(name, args, session_id=session_id)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")


# ── Users / Organizations / Quota ────────────────────────────


@router.get("/users")
async def list_users() -> list[dict[str, Any]]:
    return get_user_store().list_users()


@router.post("/users")
async def upsert_user(request: UserUpsertRequest) -> dict[str, Any]:
    try:
        return get_user_store().upsert_user(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/users/{user_id}")
async def get_user(user_id: str) -> dict[str, Any]:
    user = get_user_store().get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    user["usage"] = get_billing_store().usage_summary(user_id=user_id)
    user["quota"] = get_quota_store().status(user_id=user_id, org_id="")
    user["quota_policy"] = get_quota_store().get_policy(scope_type="user", scope_id=user_id)
    return user


@router.get("/organizations")
async def list_organizations() -> list[dict[str, Any]]:
    return get_user_store().list_organizations()


@router.get("/organizations/{org_id}")
async def get_organization(org_id: str) -> dict[str, Any]:
    org = get_user_store().get_organization(org_id)
    if org is None:
        raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found")
    org["usage"] = get_billing_store().usage_summary(org_id=org_id)
    org["quota"] = get_quota_store().status(user_id="", org_id=org_id)
    return org


@router.post("/organizations")
async def upsert_organization(request: OrganizationUpsertRequest) -> dict[str, Any]:
    try:
        return get_user_store().upsert_organization(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/organization-members")
async def add_organization_member(request: OrganizationMemberRequest) -> dict[str, str]:
    get_user_store().add_member(user_id=request.user_id, org_id=request.org_id, role=request.role)
    return {"status": "ok", "user_id": request.user_id, "org_id": request.org_id, "role": request.role}


@router.get("/quota/policies")
async def list_quota_policies() -> list[dict[str, Any]]:
    return get_quota_store().list_policies()


@router.get("/quota/policies/{scope_type}/{scope_id:path}")
async def get_quota_policy(scope_type: str, scope_id: str = "") -> dict[str, Any]:
    policy = get_quota_store().get_policy(scope_type=scope_type, scope_id=scope_id if scope_type != "default" else "")
    if policy is None:
        raise HTTPException(status_code=404, detail=f"Quota policy '{scope_type}:{scope_id}' not found")
    return policy


@router.get("/quota/policies/default")
async def get_default_quota_policy() -> dict[str, Any]:
    policy = get_quota_store().get_policy(scope_type="default", scope_id="")
    if policy is None:
        raise HTTPException(status_code=404, detail="Default quota policy not found")
    return policy


@router.post("/quota/policies")
async def upsert_quota_policy(request: QuotaPolicyRequest) -> dict[str, Any]:
    try:
        return get_quota_store().upsert_policy(request.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/quota/status")
async def quota_status(user_id: str = "", org_id: str = "") -> dict[str, Any]:
    if not user_id:
        ctx = get_user_store().get_user_context()
        user_id = ctx["user"]["id"]
        org_id = org_id or ctx["organization"]["id"]
    return get_quota_store().status(user_id=user_id, org_id=org_id)


# ── Session CRUD ─────────────────────────────────────────────


@router.get("/sessions")
async def list_sessions(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """列出所有会话。"""
    store = get_session_store()
    sessions = store.list_sessions(limit=limit, offset=offset)
    summaries = get_billing_store().session_summaries([s["id"] for s in sessions])
    for session in sessions:
        session["billing"] = summaries.get(session["id"])
        if session.get("user_id"):
            session["quota"] = get_quota_store().status(user_id=session["user_id"], org_id=session.get("org_id") or "")
    return sessions


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """获取会话元信息。"""
    store = get_session_store()
    info = store.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return info


@router.post("/sessions")
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    """创建新会话。"""
    store = get_session_store()
    ctx = get_user_store().get_user_context()
    metadata = {**(request.metadata or {}), "user_id": ctx["user"]["id"], "org_id": ctx["organization"]["id"]}
    sid = store.create_session(
        session_id=request.session_id,
        metadata=metadata,
        user_id=ctx["user"]["id"],
        org_id=ctx["organization"]["id"],
    )
    info = store.get_session(sid)
    return info


@router.put("/sessions/{session_id}")
async def update_session(session_id: str, request: UpdateSessionRequest) -> dict[str, Any]:
    """更新会话元信息。"""
    store = get_session_store()
    if request.metadata:
        store.update_session_metadata(session_id, request.metadata)
    info = store.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return info


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """删除会话及其消息。"""
    store = get_session_store()
    ok = store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    get_billing_store().delete_session_usage(session_id)
    return {"status": "deleted", "session_id": session_id}


# ── Message CRUD ─────────────────────────────────────────────


@router.get("/sessions/{session_id}/messages")
async def list_messages(session_id: str) -> list[dict[str, Any]]:
    """获取会话的所有消息。"""
    store = get_session_store()
    messages = store.load_messages(session_id)
    result = []
    for i, msg in enumerate(messages):
        blocks = []
        for b in msg.content:
            from agent_core.session.store import serialize_block
            blocks.append(serialize_block(b))
        result.append({
            "index": i,
            "role": msg.role,
            "content": blocks,
            "metadata": msg.metadata,
        })
    return result


@router.delete("/sessions/{session_id}/messages")
async def clear_messages(session_id: str) -> dict[str, str]:
    """清空会话的所有消息。"""
    store = get_session_store()
    store.save_messages(session_id, [], start_turn=0)
    clear_system_prompt_cache()
    return {"status": "cleared", "session_id": session_id}


# ── Session Memory ───────────────────────────────────────────


@router.get("/sessions/{session_id}/memory")
async def get_session_memory(session_id: str) -> dict[str, Any]:
    """读取会话的 durable session memory。"""
    manager = SessionMemoryManager(session_id)
    content = manager.get_memory_content()
    return {
        "session_id": session_id,
        "path": str(manager.memory_path),
        "exists": content is not None,
        "is_template_only": manager.is_template_only(content) if content is not None else False,
        "content": content or "",
    }


# ── Model Pricing CRUD ───────────────────────────────────────


@router.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """列出模型价格表。"""
    return get_billing_store().list_models()


@router.post("/models")
async def create_model(request: ModelPricingRequest) -> dict[str, Any]:
    """创建或更新一个模型价格项。"""
    try:
        return get_billing_store().upsert_model(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/models/{model_id}")
async def get_model(model_id: str, raw: bool = False) -> Any:
    """获取模型价格项。raw=True 时返回 JSON 文本，便于 Admin 编辑器复用。"""
    model = get_billing_store().get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if raw:
        return PlainTextResponse(_model_pricing_to_text(model), media_type="text/plain")
    return model


@router.put("/models/{model_id}/raw")
async def update_model_raw(model_id: str, request: Request) -> dict[str, Any]:
    """通过 JSON 文本更新模型价格项。"""
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Model pricing must be a JSON object")
        data["model_id"] = str(data.get("model_id") or model_id)
        if data["model_id"] != model_id:
            raise ValueError("model_id in body must match URL")
        return get_billing_store().upsert_model(data)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/models/{model_id}")
async def update_model(model_id: str, request: ModelPricingRequest) -> dict[str, Any]:
    """更新模型价格项。"""
    if request.model_id != model_id:
        raise HTTPException(status_code=400, detail="model_id in body must match URL")
    try:
        return get_billing_store().upsert_model(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict[str, str]:
    """删除模型价格项。历史 usage 不会被删除。"""
    ok = get_billing_store().delete_model(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {"status": "deleted", "model_id": model_id}


@router.get("/models/{model_id}/preview")
async def preview_model(model_id: str) -> dict[str, str]:
    model = get_billing_store().get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {"name": model_id, "rendered": _model_pricing_preview(model)}


# ── Runtime Config ──────────────────────────────────────────


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """获取当前运行时配置。"""
    selection = env_model_selection()
    available = configured_provider_names()

    compact_cfg = AutoCompactConfig()
    runtime_cfg = AgentRuntimeConfig()
    user_ctx = get_user_store().get_user_context()

    return {
        "model_provider": selection["model_provider"],
        "model_id": selection["model_id"],
        "codex_credentials": codex_credential_source(),
        "session_id": os.getenv("AGENT_SESSION_ID", "debug-session"),
        "model_stream_idle_timeout_seconds": runtime_cfg.model_stream_idle_timeout_seconds,
        "available_providers": sorted(available),
        "hook_events": [e.value for e in HookEvent],
        "session_memory": asdict(SessionMemoryConfig.from_env()),
        "compact": {
            "context_window": compact_cfg.context_window,
            "reserved_for_output": compact_cfg.reserved_for_output,
            "effective_context_window": compact_cfg.effective_context_window,
            "auto_compact_threshold": compact_cfg.auto_compact_threshold,
            "buffer_tokens": compact_cfg.buffer_tokens,
            "max_consecutive_failures": compact_cfg.max_consecutive_failures,
            "stream_idle_timeout_seconds": compact_cfg.compact_stream_idle_timeout_seconds,
            "enable_micro": compact_cfg.enable_micro,
            "enable_full": compact_cfg.enable_full,
        },
        "system_prompt_context": {
            "claude_md_enabled": not (_env_truthy("CLAUDE_CODE_DISABLE_CLAUDE_MDS") or _env_truthy("AGENT_DISABLE_CLAUDE_MDS")),
            "git_status_enabled": not _env_truthy("AGENT_DISABLE_GIT_STATUS"),
            "git_status_max_chars": 2000,
        },
        "permissions": {
            "mode": get_session_permission_state(os.getenv("AGENT_SESSION_ID", "debug-session")).mode,
            "mode_title": permission_mode_title(get_session_permission_state(os.getenv("AGENT_SESSION_ID", "debug-session")).mode),
            "available_modes": USER_PERMISSION_MODE_CHOICES,
        },
        "user": user_ctx["user"],
        "organization": user_ctx["organization"],
        "quota": get_quota_store().status(user_id=user_ctx["user"]["id"], org_id=user_ctx["organization"]["id"]),
        "usage": get_billing_store().usage_summary(user_id=user_ctx["user"]["id"]),
        "sandbox": get_sandbox_manager().status(),
    }


@router.get("/integrations/codex")
async def get_codex_status() -> dict[str, Any]:
    return codex_credential_source()


@router.put("/integrations/codex")
async def put_codex_credentials(request: CodexCredentialsRequest) -> dict[str, Any]:
    credential_type = str(request.credential_type or "api_key").strip().lower()
    if credential_type not in {"api_key", "access_token", "auth_json"}:
        raise HTTPException(status_code=400, detail="credential_type must be api_key, access_token, or auth_json")
    try:
        credentials = save_codex_credentials(
            secret=request.secret,
            credential_type=credential_type,
            base_url=request.base_url,
            model=request.model,
            image_model=request.image_model,
            metadata={**request.metadata, "source": request.metadata.get("source") or "admin-ui"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    clear_image_generation_client_cache()
    return {"codex": codex_credential_source(), "source": credentials.source, "auth_mode": credentials.auth_mode}


@router.delete("/integrations/codex")
async def delete_codex_integration() -> dict[str, Any]:
    deleted = delete_codex_credentials()
    clear_image_generation_client_cache()
    return {"deleted": deleted, "codex": codex_credential_source()}


@router.post("/integrations/codex/login")
async def post_codex_login(request: CodexLoginRequest) -> dict[str, Any]:
    try:
        return start_codex_login_flow(flow=request.flow)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/integrations/codex/login")
async def get_codex_login() -> dict[str, Any]:
    return codex_login_status()


@router.get("/sandbox")
async def get_sandbox() -> dict[str, Any]:
    """获取 Bash 沙箱状态与配置。"""
    return get_sandbox_manager().status()


@router.put("/sandbox")
async def update_sandbox(request: Request) -> dict[str, Any]:
    """更新 Bash 沙箱配置。请求体为 JSON，支持 snake_case 与 Claude 风格 camelCase 字段。"""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Sandbox config must be a JSON object")
        sandbox = get_sandbox_manager()
        sandbox.update_raw(data)
        return sandbox.status()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/sandbox/settings")
async def patch_sandbox_settings(request: Request) -> dict[str, Any]:
    """局部更新 Bash 沙箱配置。"""
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Sandbox settings patch must be a JSON object")
        sandbox = get_sandbox_manager()
        sandbox.set_settings(**data)
        return sandbox.status()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Hooks CRUD ──────────────────────────────────────────────


@router.get("/hooks")
async def list_hooks() -> list[str]:
    """列出所有 hooks 模板。"""
    store = get_hook_store()
    return store.list_templates()


@router.get("/hooks/{name}")
async def get_hooks(name: str = "default") -> dict[str, Any]:
    """获取 hooks 配置。"""
    store = get_hook_store()
    try:
        return store.get_raw(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Hooks template '{name}' not found")


@router.put("/hooks/{name}/raw")
async def update_hooks_raw(name: str, request: Request) -> dict[str, Any]:
    """通过原始 YAML 文本更新 hooks 配置。"""
    import yaml
    store = get_hook_store()
    try:
        body = await request.body()
        text = body.decode("utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Hooks config must be a YAML mapping")
        store.update_raw(name, data)
        return {"status": "updated", "name": name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Hooks template '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hooks")
async def create_hooks(request: Request) -> dict[str, Any]:
    """创建新 hooks 模板。"""
    import yaml
    store = get_hook_store()
    try:
        body = await request.body()
        text = body.decode("utf-8")
        data = yaml.safe_load(text) if text.strip() else None
        name = data.get("metadata", {}).get("name", "unnamed") if data else "unnamed"
        store.create_template(name, data)
        return {"status": "created", "name": name}
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/hooks/{name}")
async def delete_hooks(name: str) -> dict[str, str]:
    """删除 hooks 模板。"""
    store = get_hook_store()
    try:
        store.delete_template(name)
        return {"status": "deleted", "name": name}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/hooks/{name}/matchers")
async def get_hooks_matchers(name: str = "default") -> list[dict[str, Any]]:
    """获取解析后的 HookMatcher 列表。"""
    store = get_hook_store()
    try:
        matchers = store.load(name)
        return [m.to_dict() for m in matchers]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Hooks template '{name}' not found")
