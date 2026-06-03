from __future__ import annotations

import os
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_core.billing.store import BillingStore
from agent_core.buddy import BuddyStore, companion_payload, get_companion
from agent_core.context.compact import AutoCompactConfig
from agent_core.memory.session_memory import SessionMemoryConfig, SessionMemoryManager
from agent_core.permissions.policy import (
    USER_PERMISSION_MODE_CHOICES,
    FilePermissionRule,
    get_session_permission_state,
    permission_mode_title,
    persist_permission_metadata,
    reset_session_permission_rules,
    set_session_permission_mode,
)
from agent_core.plan_mode import clear_plan_slug, get_plan_file_path, persist_plan_metadata, prepare_context_for_plan_mode
from agent_core.quota.store import QuotaStore
from agent_core.sandbox import get_sandbox_manager
from agent_core.skills.store import SkillStore
from agent_core.session.store import SessionStore
from agent_core.users.store import UserStore
from agent_server.model_providers import available_provider_names, default_model_for_provider, detect_provider_from_env
from agent_server.prompt_store import PromptStore, clear_system_prompt_cache


CommandRunningChecker = Callable[[str], Awaitable[bool]]
CommandAborter = Callable[[str], Awaitable[bool]]
CommandCompactor = Callable[[str, str | None], Awaitable[dict[str, Any]]]


def command_specs() -> list[dict[str, str]]:
    return [
        {"name": "/help", "usage": "/help", "description": "显示可用命令"},
        {"name": "/status", "usage": "/status", "description": "显示当前 session 与运行状态"},
        {"name": "/clear", "usage": "/clear", "description": "清空当前 session 的聊天历史并重置终端显示"},
        {"name": "/compact", "usage": "/compact [summary instructions]", "description": "压缩当前 session 历史，保留摘要继续对话"},
        {"name": "/plan", "usage": "/plan", "description": "进入 Plan Mode；只读探索并把计划写入 plan 文件，审批后再实现"},
        {"name": "/memory", "usage": "/memory [show|path|clear]", "description": "查看或清理当前 session memory"},
        {"name": "/skills", "usage": "/skills [reload|show <name>]", "description": "列出、刷新或查看本地 skills"},
        {"name": "/model", "usage": "/model [list|current|<model_id>|<provider> <model_id>]", "description": "查看或切换当前 session 使用的模型"},
        {"name": "/prompt", "usage": "/prompt [list|current|<template>]", "description": "查看或切换当前 session 使用的 Prompt 模板"},
        {"name": "/permissions", "usage": "/permissions [mode|reset|allow-bash|revoke-bash|allow-skill|revoke-skill|allow-write|revoke-write|allow-web|revoke-web|allow-web-search|revoke-web-search]", "description": "查看/切换权限模式并管理当前 session 权限规则"},
        {"name": "/yolo", "usage": "/yolo", "description": "切换到 YOLO 权限模式（等价于 /permissions yolo）"},
        {"name": "/sandbox", "usage": "/sandbox [enable|disable|open|closed|auto-on|auto-off|exclude <pattern>|unexclude <pattern>]", "description": "查看或配置 Bash 沙箱"},
        {"name": "/quota", "usage": "/quota", "description": "查看当前用户、组织、额度与用量"},
        {"name": "/buddy", "usage": "/buddy [status|pet|mute|unmute]", "description": "唤醒、查看或互动你的终端伙伴"},
        {"name": "/config", "usage": "/config", "description": "显示模型 provider 与 memory 配置"},
        {"name": "/abort", "usage": "/abort", "description": "中断当前运行中的 agent task"},
    ]


async def handle_slash_command(
    line: str,
    *,
    session_id: str,
    session_store: SessionStore,
    is_running: CommandRunningChecker,
    abort: CommandAborter,
    compact: CommandCompactor | None = None,
    user_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a terminal slash command and return a JSON UI event."""
    parts = line.strip().split()
    command = parts[0].lower() if parts else "/help"
    args = parts[1:]

    if command in ("/help", "/?"):
        return _result(
            "Slash Commands",
            "\n".join(f"{spec['usage']:<24} {spec['description']}" for spec in command_specs()),
            commands=command_specs(),
        )

    if command == "/status":
        running = await is_running(session_id)
        return _result(
            "Status",
            f"session_id: {session_id}\nrunning: {str(running).lower()}",
            running=running,
            session_id=session_id,
        )

    if command == "/abort":
        ok = await abort(session_id)
        return _result("Abort", "abort signal sent" if ok else "no running task", running=False)

    if command == "/clear":
        if await is_running(session_id):
            return _error("Clear failed", "agent is still running; use Ctrl+C or /abort first")
        session_store.save_messages(session_id, [], start_turn=0)
        clear_system_prompt_cache()
        clear_plan_slug(session_id)
        persist_plan_metadata(session_store, session_id, cwd=_command_cwd())
        return _result(
            "Cleared",
            "Current session messages and cached prompt context were cleared. Session memory was kept.",
            clear_terminal=True,
        )

    if command == "/plan":
        _, previous = prepare_context_for_plan_mode(session_id)
        plan_file = get_plan_file_path(session_id, cwd=_command_cwd())
        persist_plan_metadata(session_store, session_id, cwd=_command_cwd())
        return _result(
            "Plan Mode",
            "Entered Plan Mode. Explore/read only, write the final plan to the plan file, then call ExitPlanMode for approval.\n\n"
            f"plan_file: {plan_file}\n"
            f"previous_mode: {previous}",
            permission_mode="plan",
            plan_file=str(plan_file),
        )

    if command == "/compact":
        if await is_running(session_id):
            return _error("Compact failed", "agent is still running; use Ctrl+C or /abort first")
        if compact is None:
            clear_system_prompt_cache()
            return _result("Compacted", "Cached prompt context was cleared.")
        try:
            result = await compact(session_id, " ".join(args).strip() or None)
        except ValueError as exc:
            return _error("Compact failed", str(exc))
        clear_system_prompt_cache()
        lines = [
            "Conversation history was compacted into a summary.",
            f"messages: {result.get('pre_message_count', '?')} → {result.get('post_message_count', '?')}",
            f"estimated_tokens: {result.get('pre_token_count', '?')} → {result.get('post_token_count', '?')}",
        ]
        summary = result.get("summary")
        if summary:
            preview = str(summary)
            if len(preview) > 1200:
                preview = preview[:1200] + "\n..."
            lines.extend(["", "Summary preview:", preview])
        return _result("Compacted", "\n".join(lines), compacted=True, **result)

    if command == "/memory":
        return await _memory_command(args, session_id=session_id, is_running=is_running)

    if command in ("/skills", "/skill"):
        return _skills_command(args, session_id=session_id)

    if command in ("/model", "/models"):
        return await _model_command(args, session_id=session_id, session_store=session_store, is_running=is_running)

    if command in ("/prompt", "/prompts"):
        return await _prompt_command(args, session_id=session_id, session_store=session_store, is_running=is_running)

    if command == "/yolo":
        args = ["yolo"]

    if command in ("/permissions", "/permission", "/mode", "/yolo"):
        return _permissions_command(args, session_id=session_id, session_store=session_store)

    if command == "/sandbox":
        return _sandbox_command(args)

    if command == "/quota":
        return _quota_command(user_context=user_context)

    if command == "/buddy":
        return _buddy_command(args, user_context=user_context)

    if command == "/config":
        provider = _detect_provider()
        selection = _session_model_selection(session_store, session_id)
        memory_cfg = asdict(SessionMemoryConfig.from_env())
        compact_cfg = asdict(AutoCompactConfig())
        lines = [
            f"model_provider: {selection['model_provider']}",
            f"model_id: {selection['model_id']}",
            f"env_default_provider: {provider}",
            f"prompt_template: {_session_prompt_template(session_store, session_id)}",
            "system_prompt_context:",
            f"  claude_md_enabled: {str(not _env_truthy('CLAUDE_CODE_DISABLE_CLAUDE_MDS') and not _env_truthy('AGENT_DISABLE_CLAUDE_MDS')).lower()}",
            f"  git_status_enabled: {str(not _env_truthy('AGENT_DISABLE_GIT_STATUS')).lower()}",
            "  git_status_max_chars: 2000",
            "permissions:",
            f"  mode: {get_session_permission_state(session_id).mode}",
            "compact:",
            f"  effective_context_window: {AutoCompactConfig().effective_context_window}",
            f"  auto_compact_threshold: {AutoCompactConfig().auto_compact_threshold}",
            f"  buffer_tokens: {compact_cfg['buffer_tokens']}",
            f"  max_consecutive_failures: {compact_cfg['max_consecutive_failures']}",
            "session_memory:",
            *[f"  {k}: {v}" for k, v in memory_cfg.items()],
        ]
        return _result(
            "Config",
            "\n".join(lines),
            config={
                "model_provider": selection["model_provider"],
                "model_id": selection["model_id"],
                "env_default_provider": provider,
                "prompt_template": _session_prompt_template(session_store, session_id),
                "session_memory": memory_cfg,
                "compact": compact_cfg,
                "permissions": {"mode": get_session_permission_state(session_id).mode},
                "sandbox": get_sandbox_manager().status(),
            },
        )

    return _error(
        "Unknown command",
        f"Unknown slash command: {command}\nType /help to see available commands.",
    )


async def _memory_command(
    args: list[str],
    *,
    session_id: str,
    is_running: CommandRunningChecker,
) -> dict[str, Any]:
    sub = (args[0].lower() if args else "show")
    manager = SessionMemoryManager(session_id)
    content = manager.get_memory_content()

    if sub in ("show", "cat", ""):
        if not content:
            return _result("Session Memory", f"No memory file yet. Expected path:\n{manager.memory_path}")
        if manager.is_template_only(content):
            return _result("Session Memory", f"Memory file exists but is still the empty template.\n\nPath: {manager.memory_path}\n\n{content}")
        return _result("Session Memory", f"Path: {manager.memory_path}\n\n{content}", memory_path=str(manager.memory_path))

    if sub == "path":
        return _result("Session Memory Path", str(manager.memory_path), memory_path=str(manager.memory_path))

    if sub in ("clear", "reset"):
        if await is_running(session_id):
            return _error("Memory clear failed", "agent is still running; use Ctrl+C or /abort first")
        removed: list[str] = []
        for path in (manager.memory_path, manager.state_path):
            with _suppress_file_not_found():
                path.unlink()
                removed.append(str(path))
        return _result(
            "Session Memory Cleared",
            "Removed:\n" + ("\n".join(removed) if removed else "nothing to remove"),
            memory_path=str(manager.memory_path),
        )

    return _error("Unknown memory command", "Usage: /memory [show|path|clear]")


def _sandbox_command(args: list[str]) -> dict[str, Any]:
    sandbox = get_sandbox_manager()
    sub = args[0].lower() if args else "status"
    settings = sandbox.settings_dict()
    try:
        if sub in ("enable", "on"):
            sandbox.set_settings(enabled=True)
        elif sub in ("disable", "off"):
            sandbox.set_settings(enabled=False)
        elif sub in ("open", "allow-unsandboxed"):
            sandbox.set_settings(allow_unsandboxed_commands=True)
        elif sub in ("closed", "strict"):
            sandbox.set_settings(allow_unsandboxed_commands=False)
        elif sub in ("auto-on", "auto"):
            sandbox.set_settings(auto_allow_bash_if_sandboxed=True)
        elif sub in ("auto-off", "manual"):
            sandbox.set_settings(auto_allow_bash_if_sandboxed=False)
        elif sub == "exclude":
            if len(args) < 2:
                return _error("Sandbox", "Usage: /sandbox exclude <command-pattern>")
            pattern = " ".join(args[1:]).strip()
            excluded = list(sandbox.settings.excluded_commands)
            if pattern not in excluded:
                excluded.append(pattern)
            sandbox.set_settings(excluded_commands=excluded)
        elif sub in ("unexclude", "include"):
            if len(args) < 2:
                return _error("Sandbox", "Usage: /sandbox unexclude <command-pattern>")
            pattern = " ".join(args[1:]).strip()
            sandbox.set_settings(excluded_commands=[p for p in sandbox.settings.excluded_commands if p != pattern])
        elif sub not in ("status", "show", ""):
            return _error("Sandbox", "Usage: /sandbox [enable|disable|open|closed|auto-on|auto-off|exclude <pattern>|unexclude <pattern>]")
    except Exception as exc:
        return _error("Sandbox update failed", str(exc))

    status = sandbox.status()
    s = status["settings"]
    fs = s["filesystem"]
    net = s["network"]
    lines = [
        f"enabled: {str(status['enabled']).lower()} (settings: {str(status['enabled_in_settings']).lower()})",
        f"platform: {status['platform']} supported: {str(status['supported']).lower()}",
        f"config_path: {status['config_path']}",
        f"auto_allow_bash_if_sandboxed: {str(s['auto_allow_bash_if_sandboxed']).lower()}",
        f"allow_unsandboxed_commands: {str(s['allow_unsandboxed_commands']).lower()}",
        f"excluded_commands: {', '.join(s['excluded_commands']) if s['excluded_commands'] else 'None'}",
        "filesystem:",
        f"  allow_write: {', '.join(fs['allow_write']) if fs['allow_write'] else 'None'}",
        f"  deny_write: {', '.join(fs['deny_write']) if fs['deny_write'] else 'None'}",
        f"  deny_read: {', '.join(fs['deny_read']) if fs['deny_read'] else 'None'}",
        f"  allow_read: {', '.join(fs['allow_read']) if fs['allow_read'] else 'None'}",
        "network:",
        f"  allowed_domains: {', '.join(net['allowed_domains']) if net['allowed_domains'] else 'None'}",
        f"  denied_domains: {', '.join(net['denied_domains']) if net['denied_domains'] else 'None'}",
    ]
    if status["errors"]:
        lines.extend(["errors:", *[f"  - {e}" for e in status["errors"]]])
    if status["warnings"]:
        lines.extend(["warnings:", *[f"  - {w}" for w in status["warnings"]]])
    return _result("Sandbox", "\n".join(lines), sandbox=status)


def _permissions_command(args: list[str], *, session_id: str, session_store: SessionStore) -> dict[str, Any]:
    state = get_session_permission_state(session_id)
    changed = False
    sub = args[0].lower() if args else "status"

    if sub in USER_PERMISSION_MODE_CHOICES or sub in {"accept-edits", "accept", "bypass", "bypass-permissions", "yolo"}:
        state = set_session_permission_mode(session_id, args[0])
        changed = True
    elif sub in ("reset", "clear"):
        state = reset_session_permission_rules(session_id, reset_mode=True)
        changed = True
    elif sub == "allow-bash":
        prefix = " ".join(args[1:]).strip()
        if not prefix:
            return _error("Permissions", "Usage: /permissions allow-bash <command-prefix>")
        state.bash_prefixes.add(prefix)
        changed = True
    elif sub == "revoke-bash":
        prefix = " ".join(args[1:]).strip()
        if not prefix:
            return _error("Permissions", "Usage: /permissions revoke-bash <command-prefix>")
        state.bash_prefixes.discard(prefix)
        changed = True
    elif sub == "allow-skill":
        skill = _normalize_permission_skill(" ".join(args[1:]))
        if not skill:
            return _error("Permissions", "Usage: /permissions allow-skill <skill-name>")
        state.skill_rules.add(skill)
        changed = True
    elif sub == "revoke-skill":
        skill = _normalize_permission_skill(" ".join(args[1:]))
        if not skill:
            return _error("Permissions", "Usage: /permissions revoke-skill <skill-name>")
        state.skill_rules.discard(skill)
        changed = True
    elif sub == "allow-write":
        path_arg = " ".join(args[1:]).strip()
        if not path_arg:
            return _error("Permissions", "Usage: /permissions allow-write <path>")
        root = _resolve_permission_path(path_arg)
        if not any(rule.operation == "write" and Path(rule.root).expanduser().resolve() == root for rule in state.file_rules):
            state.file_rules.append(FilePermissionRule(root=str(root), operation="write", scope="manual"))
        changed = True
    elif sub == "revoke-write":
        path_arg = " ".join(args[1:]).strip()
        if not path_arg:
            return _error("Permissions", "Usage: /permissions revoke-write <path>")
        root = _resolve_permission_path(path_arg)
        state.file_rules = [
            rule for rule in state.file_rules
            if not (rule.operation == "write" and Path(rule.root).expanduser().resolve() == root)
        ]
        changed = True
    elif sub == "allow-web":
        domain = _normalize_permission_domain(" ".join(args[1:]))
        if not domain:
            return _error("Permissions", "Usage: /permissions allow-web <domain-or-url>")
        state.web_domains.add(domain)
        changed = True
    elif sub == "revoke-web":
        domain = _normalize_permission_domain(" ".join(args[1:]))
        if not domain:
            return _error("Permissions", "Usage: /permissions revoke-web <domain-or-url>")
        state.web_domains.discard(domain)
        changed = True
    elif sub == "allow-web-search":
        state.web_search_allowed = True
        changed = True
    elif sub == "revoke-web-search":
        state.web_search_allowed = False
        changed = True
    elif sub not in ("status", "show", "list", "ls", ""):
        return _error("Permissions", "Usage: /permissions [default|acceptEdits|plan|bypassPermissions|yolo|reset|allow-bash <prefix>|revoke-bash <prefix>|allow-skill <name>|revoke-skill <name>|allow-write <path>|revoke-write <path>|allow-web <domain>|revoke-web <domain>|allow-web-search|revoke-web-search]")

    if changed:
        persist_permission_metadata(session_store, session_id)
        persist_plan_metadata(session_store, session_id, cwd=_command_cwd())

    lines = [
        f"permission_mode: {state.mode} ({permission_mode_title(state.mode)})",
        f"session_bash_rules: {len(state.bash_prefixes)}",
        *[f"  Bash({prefix}:*)" for prefix in sorted(state.bash_prefixes)],
        f"session_skill_rules: {len(state.skill_rules)}",
        *[f"  Skill({rule})" for rule in sorted(state.skill_rules)],
        f"session_file_rules: {len(state.file_rules)}",
        *[f"  {rule.operation} {rule.root} ({rule.scope})" for rule in state.file_rules],
        f"session_web_domain_rules: {len(state.web_domains)}",
        *[f"  WebFetch({domain})" for domain in sorted(state.web_domains)],
        f"session_web_search_allowed: {str(state.web_search_allowed).lower()}",
        "",
        "management:",
        "  /permissions reset                    reset mode to default and clear session rules",
        "  /permissions allow-bash <prefix>      allow Bash(<prefix>:*) for this session",
        "  /permissions revoke-bash <prefix>     remove a bash prefix rule",
        "  /permissions allow-skill <name>       allow Skill(<name>) for this session",
        "  /permissions revoke-skill <name>      remove a skill rule",
        "  /permissions allow-write <path>       allow writes under a file/directory path",
        "  /permissions revoke-write <path>      remove a write rule",
        "  /permissions allow-web <domain>       allow WebFetch for a domain",
        "  /permissions revoke-web <domain>      remove a WebFetch domain rule",
        "  /permissions allow-web-search         allow WebSearch for this session",
        "  /permissions revoke-web-search        remove WebSearch session approval",
        "",
        "modes:",
        "  default           ask before risky tools; read/search and safe bash are allowed",
        "  acceptEdits       auto-allow project file edits; still ask for bash/sensitive paths",
        "  plan              allow planning/read tools only; deny mutating tools and bash",
        "  bypassPermissions allow all tools without prompting",
        "  yolo              alias of bypassPermissions; explicit full-permission mode",
        "",
        f"available_modes: {', '.join(USER_PERMISSION_MODE_CHOICES)}",
    ]
    return _result(
        "Permissions",
        "\n".join(lines),
        permission_mode=state.mode,
        permission_rules={
            "bash_prefixes": sorted(state.bash_prefixes),
            "skill_rules": sorted(state.skill_rules),
            "file_rules": [{"root": rule.root, "operation": rule.operation, "scope": rule.scope} for rule in state.file_rules],
            "web_domains": sorted(state.web_domains),
            "web_search_allowed": state.web_search_allowed,
        },
    )


def _resolve_permission_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = _command_cwd() / path
    return path.resolve()


def _normalize_permission_skill(raw: str) -> str:
    skill = raw.strip()
    return skill[1:] if skill.startswith("/") else skill


def _normalize_permission_domain(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return ""
    if "://" in value:
        try:
            value = urllib.parse.urlparse(value).hostname or ""
        except Exception:
            return ""
    return value.removeprefix("www.").strip()


def _quota_command(*, user_context: dict[str, Any] | None = None) -> dict[str, Any]:
    users = UserStore()
    quota = QuotaStore()
    billing = BillingStore()
    ctx = user_context or users.get_user_context()
    user = ctx["user"]
    org = ctx["organization"]
    status = quota.status(user_id=user["id"], org_id=org["id"])
    usage = billing.usage_summary(user_id=user["id"])
    limited = [c for c in status.get("checks", []) if c.get("limit", 0) > 0]
    bars = [_quota_bar_payload(check) for check in limited]
    lines = [
        f"user: {user['id']} account_uuid: {user.get('account_uuid')}",
        f"organization: {org['id']} organization_uuid: {org.get('organization_uuid')}",
        f"subscription_type: {user.get('subscription_type') or '-'} rate_limit_tier: {user.get('rate_limit_tier') or '-'}",
        f"quota_enabled: {str(status.get('enabled')).lower()}",
        "usage:",
        f"  requests: {usage.get('request_count', 0)}",
        f"  tokens: {usage.get('total_tokens', 0)}",
        f"  cost: {usage.get('currency', 'CNY')} {usage.get('total_cost', 0):.6f}",
    ]
    if limited:
        lines.append("quota progress:")
        for bar in bars:
            lines.append(
                f"  {bar['label']:<28} {bar['bar']} "
                f"{bar['percent_text']:>7}  {bar['used_text']}/{bar['limit_text']}  remaining={bar['remaining_text']}"
            )
    else:
        lines.append("limits: unlimited (all configured limits are 0)")
    return _result(
        "Quota",
        "\n".join(lines),
        user=user,
        organization=org,
        quota=status,
        usage=usage,
        quota_bars=bars,
    )


def _buddy_command(args: list[str], *, user_context: dict[str, Any] | None = None) -> dict[str, Any]:
    users = UserStore()
    ctx = user_context or users.get_user_context()
    user = ctx["user"]
    org = ctx["organization"]
    sub = (args[0].lower() if args else "status")
    store = BuddyStore()

    if sub in ("mute", "off"):
        companion = get_companion(ctx, create=True)
        store.set_muted(user_id=user["id"], muted=True)
        companion = get_companion(ctx, create=False)
        return _result(
            "Buddy muted",
            f"{companion.name if companion else 'Buddy'} 会继续待在这里，但暂时不说话、不注入 prompt。",
            buddy_action="mute",
            companion=companion_payload(companion),
        )

    if sub in ("unmute", "on"):
        companion = get_companion(ctx, create=True)
        store.set_muted(user_id=user["id"], muted=False)
        companion = get_companion(ctx, create=False)
        return _result(
            "Buddy unmuted",
            f"{companion.name if companion else 'Buddy'} 回来了。",
            buddy_action="unmute",
            companion=companion_payload(companion),
        )

    if sub == "pet":
        companion = get_companion(ctx, create=True)
        return _result(
            "Buddy pet",
            f"你摸了摸 {companion.name}。它看起来更亮了一点。",
            buddy_action="pet",
            companion=companion_payload(companion),
        )

    if sub not in ("status", "show", "hatch", ""):
        return _error("Buddy", "Usage: /buddy [status|pet|mute|unmute]")

    companion = get_companion(ctx, create=True)
    stats = "  ".join(f"{k}={v}" for k, v in companion.stats.items())
    lines = [
        f"name: {companion.name}",
        f"species: {companion.species}  rarity: {companion.rarity}  shiny: {str(companion.shiny).lower()}",
        f"personality: {companion.personality}",
        f"face: {companion_payload(companion)['face']}",
        f"muted: {str(companion.muted).lower()}",
        f"user: {user['id']}  organization: {org['id']}",
        "stats:",
        f"  {stats}",
        "",
        "try: /buddy pet, /buddy mute, /buddy unmute",
    ]
    return _result(
        "Buddy",
        "\n".join(lines),
        buddy_action="status",
        companion=companion_payload(companion),
    )


def _quota_bar_payload(check: dict[str, Any]) -> dict[str, Any]:
    used = float(check.get("used") or 0)
    limit = float(check.get("limit") or 0)
    ratio = 0.0 if limit <= 0 else min(1.0, max(0.0, used / limit))
    percent = ratio * 100
    filled = int(round(ratio * 20))
    bar = "[" + "█" * filled + "░" * (20 - filled) + "]"
    remaining = check.get("remaining")
    status = "exceeded" if limit > 0 and used >= limit else "warn" if percent >= 80 else "ok"
    metric = str(check.get("metric") or "")
    return {
        "scope_type": check.get("scope_type"),
        "scope_id": check.get("scope_id"),
        "window": check.get("window"),
        "metric": metric,
        "label": f"{check.get('scope_type')} {check.get('window')} {metric}",
        "used": check.get("used"),
        "limit": check.get("limit"),
        "remaining": remaining,
        "percent": round(percent, 2),
        "percent_text": f"{percent:.1f}%",
        "used_text": _format_quota_value(check.get("used"), metric),
        "limit_text": _format_quota_value(check.get("limit"), metric),
        "remaining_text": "unlimited" if remaining is None else _format_quota_value(remaining, metric),
        "bar": bar,
        "status": status,
    }


def _format_quota_value(value: Any, metric: str) -> str:
    if value is None:
        return "-"
    if metric == "cost":
        return f"{float(value):.4f}"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _skills_command(args: list[str], *, session_id: str) -> dict[str, Any]:
    store = SkillStore(cwd=Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve())
    sub = args[0].lower() if args else "list"
    if sub in ("reload", "refresh"):
        store.clear_cache()
        clear_system_prompt_cache()
        skills = store.list_skills(refresh=True)
        return _result("Skills Reloaded", _format_skill_list(skills), skills=[_skill_json(s) for s in skills])
    if sub in ("show", "cat", "view"):
        if len(args) < 2:
            return _error("Skill show failed", "Usage: /skills show <name>")
        name = args[1]
        try:
            content = store.render_skill(name, " ".join(args[2:]), session_id=session_id)
        except FileNotFoundError as exc:
            return _error("Skill not found", str(exc))
        return _result(f"Skill: {name}", content, skill=name)
    skills = store.list_skills()
    return _result("Skills", _format_skill_list(skills), skills=[_skill_json(s) for s in skills])


async def _model_command(
    args: list[str],
    *,
    session_id: str,
    session_store: SessionStore,
    is_running: CommandRunningChecker,
) -> dict[str, Any]:
    billing_store = BillingStore()
    sub = args[0].lower() if args else "current"

    if sub in ("list", "ls"):
        selection = _session_model_selection(session_store, session_id)
        _ensure_current_model_in_table(billing_store, selection)
        models = billing_store.list_models()
        return _result(
            "模型列表",
            _format_model_list(models, current_model_id=selection["model_id"]),
            models=models,
            current_model=selection,
        )

    if sub in ("current", "show", "status", ""):
        selection = _session_model_selection(session_store, session_id)
        _ensure_current_model_in_table(billing_store, selection)
        model = billing_store.get_model(selection["model_id"])
        return _result(
            "当前模型",
            _format_current_model(selection, model),
            current_model=selection,
            model=model,
        )

    if await is_running(session_id):
        return _error("模型切换失败", "agent 正在运行中；请先 Ctrl+C 或执行 /abort 后再切换模型")

    provider, model_id = _parse_model_switch_args(args, billing_store=billing_store, current=_session_model_selection(session_store, session_id))
    metadata = _merged_session_metadata(
        session_store,
        session_id,
        {
            "model_provider": provider,
            "model_id": model_id,
        },
    )
    info = session_store.get_session(session_id)
    if info is None:
        session_store.create_session(session_id=session_id, metadata=metadata)
    else:
        session_store.update_session_metadata(session_id, metadata)

    selection = _session_model_selection(session_store, session_id)
    _ensure_current_model_in_table(billing_store, selection)
    model = billing_store.get_model(selection["model_id"])
    lines = [
        "当前 session 后续对话已切换模型。",
        "",
        _format_current_model(selection, model),
        "",
        "提示：模型切换只影响当前 session，已保存到 session metadata；历史消息和账单不会被清空。",
    ]
    return _result(
        "模型已切换",
        "\n".join(lines),
        model_changed=True,
        current_model=selection,
        model=model,
    )


async def _prompt_command(
    args: list[str],
    *,
    session_id: str,
    session_store: SessionStore,
    is_running: CommandRunningChecker,
) -> dict[str, Any]:
    store = PromptStore()
    sub = args[0].strip() if args else "current"
    current = _session_prompt_template(session_store, session_id)
    templates = store.list_templates()

    if sub.lower() in ("list", "ls"):
        return _result("Prompt 模板列表", _format_prompt_template_list(templates, current=current), templates=templates, current_prompt_template=current)

    if sub.lower() in ("current", "show", "status", ""):
        return _result("当前 Prompt 模板", f"prompt_template: {current}\n\n使用 /prompt list 查看可选模板；使用 /prompt <template> 切换当前 session。", current_prompt_template=current)

    if await is_running(session_id):
        return _error("Prompt 切换失败", "agent 正在运行中；请先 Ctrl+C 或执行 /abort 后再切换 Prompt 模板")

    template_name = sub
    try:
        store.get_template(template_name)
    except FileNotFoundError:
        return _error("Prompt 模板不存在", f"找不到 Prompt 模板：{template_name}\n\n可执行 /prompt list 查看可选模板。")

    metadata = _merged_session_metadata(session_store, session_id, {"prompt_template": template_name})
    info = session_store.get_session(session_id)
    if info is None:
        session_store.create_session(session_id=session_id, metadata=metadata)
    else:
        session_store.update_session_metadata(session_id, metadata)
    clear_system_prompt_cache()
    return _result(
        "Prompt 已切换",
        f"当前 session 后续对话将使用 Prompt 模板：{template_name}\n\n提示：只影响新一轮运行时构建；历史消息不会清空。",
        prompt_changed=True,
        current_prompt_template=template_name,
    )


def _parse_model_switch_args(args: list[str], *, billing_store: BillingStore, current: dict[str, str]) -> tuple[str, str]:
    providers = available_provider_names()
    first = args[0].strip()
    if len(args) >= 2 and first.lower() in providers:
        return first.lower(), args[1].strip()
    if len(args) == 1 and first.lower() in providers:
        provider = first.lower()
        # 只切 provider 时，沿用 runtime_factory 中的默认模型解析逻辑。
        return provider, default_model_for_provider(provider)

    model_id = first
    model = billing_store.get_model(model_id)
    if model_id in {"fake", "fake_tool", "fake_thinking"}:
        return model_id, model_id
    if model and model.get("provider"):
        return str(model["provider"]).lower(), model_id
    return current["model_provider"], model_id


def _ensure_current_model_in_table(billing_store: BillingStore, selection: dict[str, str]) -> dict[str, Any] | None:
    """把当前 runtime/session 模型补齐进模型表，保证所有模型视图读同一张表。"""
    model_id = selection.get("model_id") or ""
    if not model_id:
        return None
    return billing_store.ensure_model(
        model_id=model_id,
        provider=selection.get("model_provider") or "",
        display_name=model_id,
    )


def _merged_session_metadata(session_store: SessionStore, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    info = session_store.get_session(session_id)
    metadata = dict(info.get("metadata") or {}) if info else {}
    metadata.update(updates)
    return metadata


def _session_model_selection(session_store: SessionStore, session_id: str) -> dict[str, str]:
    metadata: dict[str, Any] = {}
    info = session_store.get_session(session_id)
    if info and isinstance(info.get("metadata"), dict):
        metadata = info["metadata"]
    provider = str(metadata.get("model_provider") or "").strip().lower()
    model_id = str(metadata.get("model_id") or "").strip()
    if not provider or provider == "auto":
        provider = _detect_provider()
    if not model_id:
        model_id = _default_model_id_for_provider(provider)
    return {"model_provider": provider, "model_id": model_id}


def _session_prompt_template(session_store: SessionStore, session_id: str) -> str:
    metadata: dict[str, Any] = {}
    info = session_store.get_session(session_id)
    if info and isinstance(info.get("metadata"), dict):
        metadata = info["metadata"]
    selected = str(metadata.get("prompt_template") or metadata.get("template_name") or "").strip()
    return selected or (os.getenv("AGENT_PROMPT_TEMPLATE") or os.getenv("PROMPT_TEMPLATE") or "default").strip() or "default"


def _default_model_id_for_provider(provider: str) -> str:
    return default_model_for_provider(provider)


def _format_current_model(selection: dict[str, str], model: dict[str, Any] | None) -> str:
    lines = [
        f"模型供应商：{selection['model_provider']}",
        f"模型 ID：{selection['model_id']}",
    ]
    if model:
        currency = model.get("currency") or "CNY"
        lines.extend([
            f"展示名称：{model.get('display_name') or '-'}",
            "计费价格（每百万 token）：",
            f"  输入：{currency} {model.get('input_per_million', 0)}",
            f"  输出：{currency} {model.get('output_per_million', 0)}",
            f"  缓存读取：{currency} {model.get('cache_read_per_million', 0)}",
            f"  缓存写入：{currency} {model.get('cache_write_per_million', 0)}",
        ])
        if _is_zero_priced_model(model):
            lines.append("提示：该模型已在模型表中自动补齐，但价格仍为 0；可在 Admin → Models 中编辑真实价格。")
    else:
        lines.extend([
            "计费价格：模型表中未找到该模型，账单费用会按 0 计算",
            "提示：执行 /model list 会自动把当前模型补齐进模型表；也可在 Admin → Models 中手动新增同名模型价格。",
        ])
    return "\n".join(lines)


def _format_prompt_template_list(templates: list[dict[str, Any]], *, current: str) -> str:
    if not templates:
        return "暂无 Prompt 模板。"
    lines = []
    for item in templates:
        name = str(item.get("name") or "")
        marker = "*" if name == current else " "
        desc = str(item.get("description") or "").strip()
        version = str(item.get("version") or "").strip()
        suffix = ""
        if version:
            suffix += f" v{version}"
        if desc:
            suffix += f" — {desc}"
        lines.append(f"{marker} {name}{suffix}")
    lines.extend(["", "切换：/prompt <template>"])
    return "\n".join(lines)


def _format_model_list(models: list[dict[str, Any]], *, current_model_id: str) -> str:
    if not models:
        return "暂无模型配置。可在 Admin → Models 中新增模型，或使用 /model <provider> <model_id> 临时切换。"
    lines = ["模型表（* 表示当前 session）："]
    for model in models:
        marker = "*" if model.get("model_id") == current_model_id else " "
        currency = model.get("currency") or "CNY"
        configured = not _is_zero_priced_model(model)
        price_text = (
            "未配置价格"
            if not configured
            else (
                f"{currency} 输入/输出/缓存读/缓存写="
                f"{model.get('input_per_million', 0)}/"
                f"{model.get('output_per_million', 0)}/"
                f"{model.get('cache_read_per_million', 0)}/"
                f"{model.get('cache_write_per_million', 0)}"
            )
        )
        source_text = " · 当前模型" if model.get("model_id") == current_model_id else ""
        lines.append(
            f"{marker} {model.get('model_id'):<34} "
            f"provider={model.get('provider') or '-':<10} "
            f"{price_text}{source_text}"
        )
    lines.extend([
        "",
        "用法：",
        "  /model                         查看当前模型",
        "  /model list                    列出模型价格表",
        "  /model <model_id>              按模型价格表推断 provider 并切换",
        "  /model <provider> <model_id>   指定 provider 和模型 ID 切换",
    ])
    return "\n".join(lines)


def _is_zero_priced_model(model: dict[str, Any]) -> bool:
    return not any(float(model.get(key) or 0) for key in (
        "input_per_million",
        "output_per_million",
        "cache_read_per_million",
        "cache_write_per_million",
    ))


def _format_skill_list(skills: list[Any]) -> str:
    if not skills:
        return "No skills found. Create .claude/skills/<name>/SKILL.md or ~/.claude/skills/<name>/SKILL.md."
    return "\n".join(
        f"{s.name:<24} {s.source:<8} {s.description} ({s.path})"
        for s in skills
    )


def _skill_json(skill: Any) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "when_to_use": skill.when_to_use,
        "source": skill.source,
        "path": str(skill.path),
        "base_dir": str(skill.base_dir),
        "aliases": skill.aliases,
        "version": skill.version,
        "model": skill.model,
        "context": skill.execution_context,
        "agent": skill.agent,
        "effort": skill.effort,
        "shell": skill.shell,
        "paths": skill.paths,
        "argument_names": skill.argument_names,
        "disable_model_invocation": skill.disable_model_invocation,
    }


def _result(title: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"type": "command_result", "title": title, "content": content, **extra}


def _error(title: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"type": "command_error", "title": title, "content": content, **extra}


def _detect_provider() -> str:
    return detect_provider_from_env()


def _command_cwd() -> Path:
    return Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class _suppress_file_not_found:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        return exc_type is FileNotFoundError
