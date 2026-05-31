from __future__ import annotations

import os
import re
import shlex
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from agent_core.sandbox import get_sandbox_manager
from agent_core.tools.base import Tool


PermissionStatus = Literal["allow", "ask", "deny"]
PermissionModeValue = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
PermissionOptionType = Literal["accept-once", "accept-session", "reject"]
USER_PERMISSION_MODE_CHOICES = ["default", "acceptEdits", "plan", "bypassPermissions", "yolo"]
PERMISSION_METADATA_KEY = "permissions"


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS = "bypassPermissions"


@dataclass(slots=True)
class PermissionDecision:
    status: PermissionStatus
    reason: str = ""
    options: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilePermissionRule:
    root: str
    operation: Literal["read", "write"]
    scope: str = "directory"

    def matches(self, path: Path, operation: Literal["read", "write"]) -> bool:
        try:
            root = Path(self.root).expanduser().resolve()
            target = path.expanduser().resolve()
        except Exception:
            return False
        if operation == "write" and self.operation != "write":
            return False
        if operation == "read" and self.operation not in ("read", "write"):
            return False
        return target == root or root in target.parents


@dataclass(slots=True)
class SessionPermissionState:
    mode: PermissionModeValue = "default"
    pre_plan_mode: PermissionModeValue | str = ""
    plan_slug: str = ""
    plan_accepted: bool = False
    plan_clear_context: bool = True
    plan_accept_mode: PermissionModeValue | str = ""
    bash_prefixes: set[str] = field(default_factory=set)
    skill_rules: set[str] = field(default_factory=set)
    file_rules: list[FilePermissionRule] = field(default_factory=list)
    web_domains: set[str] = field(default_factory=set)
    web_search_allowed: bool = False


_SESSION_STATES: dict[str, SessionPermissionState] = {}

READ_TOOLS = {"read_text_file", "list_directory", "grep", "glob"}
WEB_TOOLS = {"WebFetch", "WebSearch"}
ALWAYS_SAFE_TOOLS = {"echo", "ToolSearch", "TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "EnterPlanMode"}
WRITE_TOOLS = {"write_text_file", "edit_file"}
FILE_TOOLS = READ_TOOLS | WRITE_TOOLS
SAFE_ENV_VARS = {"CI", "TERM", "NO_COLOR", "FORCE_COLOR", "PYTHONPATH", "PATH", "HOME", "PWD", "LANG", "LC_ALL"}
READ_ONLY_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "grep", "rg", "find", "tree", "stat", "file", "du", "echo", "printf",
    "git status", "git diff", "git log", "git show", "git branch", "git rev-parse",
}
DANGEROUS_COMMAND_RE = re.compile(
    r"(^|\s)(rm|rmdir|unlink|dd|mkfs|mount|umount|chmod|chown|sudo|doas|pkexec)\b|"
    r"git\s+(push|reset\s+--hard|clean|rebase|checkout|switch|restore)\b|"
    r"(>|>>|\btee\b)|"
    r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh|python|ruby|perl)",
    re.IGNORECASE,
)
BARE_SHELL_PREFIXES = {
    "sh", "bash", "zsh", "fish", "csh", "tcsh", "ksh", "dash", "cmd", "powershell", "pwsh",
    "env", "xargs", "nice", "stdbuf", "nohup", "timeout", "time", "sudo", "doas", "pkexec",
}
ENV_VAR_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")


def get_session_permission_state(session_id: str | None) -> SessionPermissionState:
    key = session_id or "default"
    return _SESSION_STATES.setdefault(key, SessionPermissionState(mode=_mode_from_env()))


def set_session_permission_mode(session_id: str | None, mode: str) -> SessionPermissionState:
    state = get_session_permission_state(session_id)
    state.mode = normalize_permission_mode(mode)
    return state


def reset_session_permission_rules(session_id: str | None, *, reset_mode: bool = False) -> SessionPermissionState:
    """Clear user-approved session permission rules for a session."""
    state = get_session_permission_state(session_id)
    state.bash_prefixes.clear()
    state.skill_rules.clear()
    state.file_rules.clear()
    state.web_domains.clear()
    state.web_search_allowed = False
    if reset_mode:
        state.mode = "default"
        state.pre_plan_mode = ""
    return state


def export_permission_metadata(session_id: str | None) -> dict[str, Any]:
    """Serialize permission mode and user-managed session rules into metadata."""
    state = get_session_permission_state(session_id)
    return {
        "version": 1,
        "mode": state.mode,
        "bash_prefixes": sorted(state.bash_prefixes),
        "skill_rules": sorted(state.skill_rules),
        "file_rules": [
            {"root": rule.root, "operation": rule.operation, "scope": rule.scope}
            for rule in state.file_rules
        ],
        "web_domains": sorted(state.web_domains),
        "web_search_allowed": state.web_search_allowed,
    }


def merge_permission_metadata(metadata: dict[str, Any] | None, session_id: str | None) -> dict[str, Any]:
    """Return metadata with current permission state under the permissions key."""
    merged = dict(metadata or {})
    permission_meta = export_permission_metadata(session_id)
    has_rules = bool(
        permission_meta["bash_prefixes"]
        or permission_meta["skill_rules"]
        or permission_meta["file_rules"]
        or permission_meta["web_domains"]
        or permission_meta["web_search_allowed"]
    )
    if permission_meta.get("mode") == "default" and not has_rules:
        merged.pop(PERMISSION_METADATA_KEY, None)
    else:
        merged[PERMISSION_METADATA_KEY] = permission_meta
    return merged


def persist_permission_metadata(store: Any, session_id: str | None) -> None:
    """Best-effort write-through of current permission state into a SessionStore."""
    if not store or not session_id:
        return
    info = store.get_session(session_id)
    if info is None:
        store.create_session(session_id=session_id)
        info = store.get_session(session_id)
    metadata = dict((info or {}).get("metadata") or {})
    store.update_session_metadata(session_id, merge_permission_metadata(metadata, session_id))


def restore_permission_state(session_id: str | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Restore permission mode and session rules from SessionStore metadata."""
    permission_meta = dict((metadata or {}).get(PERMISSION_METADATA_KEY) or {})
    if not permission_meta:
        legacy_keys = ("permission_mode", "bash_prefixes", "skill_rules", "file_rules", "web_domains", "web_search_allowed")
        if not any(key in (metadata or {}) for key in legacy_keys):
            return {"restored": False, "bashRules": 0, "skillRules": 0, "fileRules": 0}
        permission_meta = {}
        if "permission_mode" in (metadata or {}):
            permission_meta["mode"] = (metadata or {}).get("permission_mode")
        for key in ("bash_prefixes", "skill_rules", "file_rules", "web_domains", "web_search_allowed"):
            if key in (metadata or {}):
                permission_meta[key] = (metadata or {}).get(key)
    state = get_session_permission_state(session_id)
    restored: dict[str, Any] = {"restored": False, "bashRules": 0, "skillRules": 0, "fileRules": 0, "webRules": 0}

    mode = permission_meta.get("mode")
    if isinstance(mode, str) and mode:
        state.mode = normalize_permission_mode(mode)
        restored["restored"] = True

    if "bash_prefixes" in permission_meta:
        prefixes = permission_meta.get("bash_prefixes")
        state.bash_prefixes = {str(prefix).strip() for prefix in prefixes if str(prefix).strip()} if isinstance(prefixes, list) else set()
        restored["bashRules"] = len(state.bash_prefixes)
        restored["restored"] = True

    if "skill_rules" in permission_meta:
        rules = permission_meta.get("skill_rules")
        state.skill_rules = {str(rule).strip().lstrip("/") for rule in rules if str(rule).strip()} if isinstance(rules, list) else set()
        restored["skillRules"] = len(state.skill_rules)
        restored["restored"] = True

    if "file_rules" in permission_meta:
        raw_rules = permission_meta.get("file_rules")
        file_rules: list[FilePermissionRule] = []
        if isinstance(raw_rules, list):
            for raw in raw_rules:
                if not isinstance(raw, dict):
                    continue
                root = str(raw.get("root") or "").strip()
                operation = str(raw.get("operation") or "write")
                scope = str(raw.get("scope") or "directory")
                if root and operation in {"read", "write"}:
                    file_rules.append(FilePermissionRule(root=root, operation=operation, scope=scope))  # type: ignore[arg-type]
        state.file_rules = file_rules
        restored["fileRules"] = len(state.file_rules)
        restored["restored"] = True
    if "web_domains" in permission_meta:
        domains = permission_meta.get("web_domains")
        state.web_domains = {_normalize_domain(str(domain)) for domain in domains if str(domain).strip()} if isinstance(domains, list) else set()
        restored["webRules"] = len(state.web_domains)
        restored["restored"] = True
    if "web_search_allowed" in permission_meta:
        state.web_search_allowed = bool(permission_meta.get("web_search_allowed"))
        restored["restored"] = True
    return restored


def normalize_permission_mode(mode: str | None) -> PermissionModeValue:
    if mode in {"default", "acceptEdits", "plan", "bypassPermissions"}:
        return mode  # type: ignore[return-value]
    aliases = {
        "accept-edits": "acceptEdits",
        "accept": "acceptEdits",
        "bypass": "bypassPermissions",
        "bypass-permissions": "bypassPermissions",
        "yolo": "bypassPermissions",
    }
    return aliases.get((mode or "").strip(), "default")  # type: ignore[return-value]


def permission_mode_title(mode: str) -> str:
    return {
        "default": "Default",
        "acceptEdits": "Accept edits",
        "plan": "Plan Mode",
        "bypassPermissions": "Bypass Permissions",
        "yolo": "YOLO",
    }.get(normalize_permission_mode(mode), "Default")


def _mode_from_env() -> PermissionModeValue:
    return normalize_permission_mode(os.getenv("AGENT_PERMISSION_MODE") or os.getenv("PERMISSION_MODE"))


class PermissionPolicy:
    """Policy/Chain of Responsibility：工具执行前统一决策。"""

    async def check(self, *, tool: Tool, tool_input: dict) -> PermissionDecision:
        return PermissionDecision(status="allow", reason="default allow")

    def record_user_decision(self, *, tool: Tool, tool_input: dict, option: dict[str, Any] | None) -> None:
        """记忆用户选择。基础策略不保存状态。"""
        return None


class StaticPermissionPolicy(PermissionPolicy):
    def __init__(
        self,
        *,
        allow: set[str] | None = None,
        deny: set[str] | None = None,
        ask: set[str] | None = None,
        session_id: str | None = None,
        mode: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._allow = allow or set()
        self._deny = deny or set()
        self._ask = ask or set()
        self._session_id = session_id or "default"
        self._cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self._state = get_session_permission_state(self._session_id)
        if mode is not None:
            self._state.mode = normalize_permission_mode(mode)

    @property
    def mode(self) -> PermissionModeValue:
        return self._state.mode

    async def check(self, *, tool: Tool, tool_input: dict) -> PermissionDecision:
        if tool.name in self._deny:
            return PermissionDecision(status="deny", reason=f"tool {tool.name} is denied by policy")

        if tool.name == "ExitPlanMode":
            if self.mode != "plan":
                return PermissionDecision(
                    status="deny",
                    reason="ExitPlanMode can only be used while Plan Mode is active.",
                )
            plan_metadata: dict[str, Any] = {"kind": "plan_approval"}
            try:
                from agent_core.plan_mode import get_plan, get_plan_file_path
                plan_metadata["planFilePath"] = str(get_plan_file_path(self._session_id, cwd=self._cwd))
                plan_metadata["plan"] = tool_input.get("plan") if isinstance(tool_input.get("plan"), str) else get_plan(self._session_id, cwd=self._cwd)
            except Exception:
                pass
            return PermissionDecision(
                status="ask",
                reason="Review and approve the plan before implementation starts.",
                options=_exit_plan_options(self._state.pre_plan_mode or "default"),
                metadata=plan_metadata,
            )

        if self.mode == "bypassPermissions":
            return PermissionDecision(status="allow", reason="allowed by bypassPermissions mode")

        if self.mode == "plan" and tool.name in WRITE_TOOLS and self._is_plan_file_write(tool_input):
            return PermissionDecision(status="allow", reason="Plan Mode allows writing the active plan file")

        if self.mode == "plan" and tool.name not in READ_TOOLS | WEB_TOOLS | ALWAYS_SAFE_TOOLS:
            return PermissionDecision(
                status="deny",
                reason="Plan Mode is active: mutating tools, bash, and subagents are disabled until the plan is approved. Write only the active plan file, then call ExitPlanMode.",
            )

        if tool.name in ALWAYS_SAFE_TOOLS or tool.name in READ_TOOLS:
            return PermissionDecision(status="allow", reason="safe read/planning tool")

        if tool.name in WEB_TOOLS:
            return self._check_web_tool(tool.name, tool_input)

        if tool.name in WRITE_TOOLS:
            return self._check_file_tool(tool.name, tool_input)

        if tool.name == "bash":
            return self._check_bash(tool_input)

        if tool.name == "Skill":
            return self._check_skill(tool_input)

        if tool.name == "Task":
            if self.mode == "plan":
                return PermissionDecision(status="deny", reason="Plan Mode is active: subagents are disabled.")
            return PermissionDecision(status="ask", reason="Subagent execution requires user approval", options=_generic_options())

        if tool.name in self._ask:
            return PermissionDecision(status="ask", reason=f"tool {tool.name} requires user approval", options=_generic_options())
        if not self._allow or tool.name in self._allow:
            return PermissionDecision(status="allow", reason="allowed by policy")
        return PermissionDecision(status="deny", reason=f"tool {tool.name} is not in allowlist")

    def record_user_decision(self, *, tool: Tool, tool_input: dict, option: dict[str, Any] | None) -> None:
        if tool.name == "ExitPlanMode":
            if option and option.get("type") != "reject":
                mode = str(option.get("mode") or option.get("permission_mode") or self._state.pre_plan_mode or "default")
                self._state.mode = normalize_permission_mode(mode)
                if self._state.mode == "plan":
                    self._state.mode = "default"
                self._state.plan_accepted = True
                self._state.plan_clear_context = bool(option.get("clearContext", True))
                self._state.plan_accept_mode = self._state.mode
                self._state.pre_plan_mode = ""
            return
        if not option or option.get("type") != "accept-session":
            return
        if tool.name in WRITE_TOOLS:
            path = _extract_path(tool_input)
            if not path:
                return
            target = Path(path).expanduser().resolve()
            scope = str(option.get("scope") or "")
            root = self._scope_root_for_path(target, scope)
            self._state.file_rules.append(FilePermissionRule(root=str(root), operation="write", scope=scope or "directory"))
        elif tool.name == "bash":
            command = str(tool_input.get("command") or tool_input.get("cmd") or "")
            prefix = get_bash_command_prefix(command)
            if prefix:
                self._state.bash_prefixes.add(prefix)
        elif tool.name == "Skill":
            skill = _normalize_skill_name(tool_input)
            if skill:
                self._state.skill_rules.add(str(option.get("skill") or skill))
        elif tool.name == "WebFetch":
            domain = _extract_url_domain(tool_input)
            if domain:
                self._state.web_domains.add(domain)
        elif tool.name == "WebSearch":
            self._state.web_search_allowed = True
        elif tool.name:
            # Generic session approvals keep parity with the existing simple ask policy.
            self._allow.add(tool.name)

    def _check_file_tool(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        path_value = _extract_path(tool_input)
        if not path_value:
            return PermissionDecision(status="ask", reason=f"{tool_name} modifies files and requires approval", options=_generic_options())
        target = self._resolve_user_path(path_value)
        if self.mode == "plan" and self._is_plan_file_path(target):
            return PermissionDecision(status="allow", reason="Plan Mode allows writing the active plan file")
        if _is_dangerous_path(target):
            return PermissionDecision(
                status="ask",
                reason=f"{tool_name} targets a sensitive path: {target}",
                options=_file_options(target, self._cwd, operation="write"),
                metadata={"risk": "sensitive_path"},
            )
        if any(rule.matches(target, "write") for rule in self._state.file_rules):
            return PermissionDecision(status="allow", reason="allowed by session file permission")
        if self.mode == "acceptEdits" and _is_within(target, self._cwd):
            return PermissionDecision(status="allow", reason="allowed by acceptEdits mode for project files")
        return PermissionDecision(
            status="ask",
            reason=f"{tool_name} wants to edit {target}",
            options=_file_options(target, self._cwd, operation="write"),
            metadata={"operation": "write", "path": str(target)},
        )

    def _check_bash(self, tool_input: dict) -> PermissionDecision:
        command = str(tool_input.get("command") or tool_input.get("cmd") or "")
        if not command.strip():
            return PermissionDecision(status="ask", reason="bash command is empty or missing", options=_generic_options())
        sandbox = get_sandbox_manager()
        if sandbox.is_auto_allow_bash_if_sandboxed_enabled() and sandbox.should_use_sandbox({**tool_input, "command": command}):
            return PermissionDecision(
                status="allow",
                reason="Auto-allowed with sandbox (auto_allow_bash_if_sandboxed enabled)",
                metadata={"sandboxed": True},
            )
        prefix = get_bash_command_prefix(command)
        if prefix and prefix in self._state.bash_prefixes:
            return PermissionDecision(status="allow", reason=f"allowed by session bash rule: {prefix}:*")
        risk = classify_bash_command(command)
        if risk == "safe":
            return PermissionDecision(status="allow", reason="safe read-only bash command")
        reason = "bash command may modify system state" if risk == "dangerous" else "bash command requires user approval"
        return PermissionDecision(status="ask", reason=reason, options=_bash_options(command), metadata={"risk": risk, "prefix": prefix})

    def _check_skill(self, tool_input: dict) -> PermissionDecision:
        skill = _normalize_skill_name(tool_input)
        if not skill:
            return PermissionDecision(status="ask", reason="Skill name is empty or missing", options=_generic_options())
        if skill in self._state.skill_rules or any(rule.endswith(":*") and skill.startswith(rule[:-2]) for rule in self._state.skill_rules):
            return PermissionDecision(status="allow", reason=f"allowed by session skill rule: {skill}")
        return PermissionDecision(
            status="ask",
            reason=f"Execute skill: {skill}",
            options=_skill_options(skill),
            metadata={"skill": skill},
        )

    def _check_web_tool(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        if tool_name == "WebSearch":
            if self._state.web_search_allowed:
                return PermissionDecision(status="allow", reason="allowed by session web search rule")
            return PermissionDecision(
                status="ask",
                reason="WebSearch may contact external search providers and requires approval",
                options=_web_search_options(),
                metadata={"operation": "web_search", "query": str(tool_input.get("query") or "")},
            )

        domain = _extract_url_domain(tool_input)
        if not domain:
            return PermissionDecision(status="ask", reason="WebFetch URL is empty or invalid", options=_generic_options())
        if domain in self._state.web_domains:
            return PermissionDecision(status="allow", reason=f"allowed by session web domain rule: {domain}")
        return PermissionDecision(
            status="ask",
            reason=f"WebFetch wants to fetch content from {domain}",
            options=_web_fetch_options(domain),
            metadata={"operation": "web_fetch", "domain": domain},
        )

    def _scope_root_for_path(self, target: Path, scope: str) -> Path:
        if scope == "claude-folder":
            return self._cwd / ".claude"
        if scope == "global-claude-folder":
            return Path.home() / ".claude"
        if _is_within(target, self._cwd):
            return self._cwd
        return target.parent

    def _resolve_user_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self._cwd / path
        return path.resolve()

    def _is_plan_file_write(self, tool_input: dict[str, Any]) -> bool:
        path_value = _extract_path(tool_input)
        if not path_value:
            return False
        return self._is_plan_file_path(self._resolve_user_path(path_value))

    def _is_plan_file_path(self, target: Path) -> bool:
        try:
            from agent_core.plan_mode import get_plan_file_path
            return target == get_plan_file_path(self._session_id, cwd=self._cwd).resolve()
        except Exception:
            return False


def _extract_path(tool_input: dict[str, Any]) -> str | None:
    value = tool_input.get("path") or tool_input.get("file_path") or tool_input.get("filepath") or tool_input.get("filename")
    return str(value) if value else None


def _normalize_skill_name(tool_input: dict[str, Any]) -> str:
    raw = str(tool_input.get("skill") or tool_input.get("name") or "").strip()
    return raw[1:] if raw.startswith("/") else raw


def _extract_url_domain(tool_input: dict[str, Any]) -> str | None:
    raw = str(tool_input.get("url") or tool_input.get("uri") or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return _normalize_domain(parsed.hostname)


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().removeprefix("www.")


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path == root or root in path.parents
    except RuntimeError:
        return False


def _is_dangerous_path(path: Path) -> bool:
    home = Path.home().resolve()
    dangerous_roots = [Path("/bin"), Path("/sbin"), Path("/usr/bin"), Path("/usr/sbin"), Path("/etc"), Path("/System")]
    if path in dangerous_roots or any(root in path.parents for root in dangerous_roots):
        return True
    sensitive_names = {".ssh", ".aws", ".gnupg", ".kube"}
    return any(part in sensitive_names for part in path.parts) or (_is_within(path, home) and path.name in {".zshrc", ".bashrc", ".bash_profile"})


def classify_bash_command(command: str) -> Literal["safe", "ask", "dangerous"]:
    stripped = command.strip()
    if not stripped:
        return "ask"
    if DANGEROUS_COMMAND_RE.search(stripped):
        return "dangerous"
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return "ask"
    tokens = _strip_safe_env(tokens)
    if not tokens:
        return "ask"
    cmd = tokens[0]
    two = " ".join(tokens[:2]) if len(tokens) >= 2 else cmd
    if two in READ_ONLY_COMMANDS or cmd in READ_ONLY_COMMANDS:
        return "safe"
    return "ask"


def get_bash_command_prefix(command: str) -> str | None:
    try:
        tokens = shlex.split(command.strip())
    except ValueError:
        tokens = command.strip().split()
    tokens = _strip_safe_env(tokens)
    if not tokens:
        return None
    first = tokens[0]
    if first in BARE_SHELL_PREFIXES or "/" in first or first.startswith("."):
        return None
    if len(tokens) >= 2 and re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", tokens[1]):
        prefix = f"{first} {tokens[1]}"
    else:
        prefix = first
    if prefix.split()[0] in BARE_SHELL_PREFIXES:
        return None
    return prefix


def _strip_safe_env(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) and ENV_VAR_ASSIGN_RE.match(tokens[i]):
        var_name = tokens[i].split("=", 1)[0]
        if var_name not in SAFE_ENV_VARS:
            return tokens
        i += 1
    return tokens[i:]


def _generic_options() -> list[dict[str, Any]]:
    return [
        {"type": "accept-once", "label": "Yes", "value": "yes"},
        {"type": "accept-session", "label": "Yes, during this session", "value": "yes-session"},
        {"type": "reject", "label": "No", "value": "no"},
    ]


def _exit_plan_options(pre_plan_mode: str) -> list[dict[str, Any]]:
    restored = normalize_permission_mode(pre_plan_mode)
    if restored == "plan":
        restored = "default"
    return [
        {"type": "accept-once", "label": f"Approve & implement ({permission_mode_title(restored)}, clear context)", "value": "yes-restore-clear-context", "mode": restored, "clearContext": True},
        {"type": "accept-once", "label": f"Approve & implement ({permission_mode_title(restored)}, keep context)", "value": "yes-restore-keep-context", "mode": restored, "clearContext": False},
        {"type": "accept-once", "label": "Approve with Accept Edits (clear context)", "value": "yes-accept-edits", "mode": "acceptEdits", "clearContext": True},
        {"type": "accept-once", "label": "Approve with Accept Edits (keep context)", "value": "yes-accept-edits-keep-context", "mode": "acceptEdits", "clearContext": False},
        {"type": "accept-once", "label": "Approve with Bypass Permissions (clear context)", "value": "yes-bypass-permissions", "mode": "bypassPermissions", "clearContext": True},
        {"type": "reject", "label": "No, keep planning", "value": "no"},
    ]


def _file_options(target: Path, cwd: Path, *, operation: Literal["read", "write"]) -> list[dict[str, Any]]:
    options = [{"type": "accept-once", "label": "Yes", "value": "yes"}]
    project_claude = cwd / ".claude"
    global_claude = Path.home() / ".claude"
    if operation != "read" and _is_within(target, project_claude):
        options.append({
            "type": "accept-session",
            "scope": "claude-folder",
            "label": "Yes, and allow edits to project .claude/ for this session",
            "value": "yes-claude-folder",
        })
    elif operation != "read" and _is_within(target, global_claude):
        options.append({
            "type": "accept-session",
            "scope": "global-claude-folder",
            "label": "Yes, and allow edits to ~/.claude/ for this session",
            "value": "yes-global-claude-folder",
        })
    else:
        label = "Yes, allow all edits during this session" if _is_within(target, cwd) else f"Yes, allow edits in {target.parent.name or str(target.parent)}/ during this session"
        options.append({"type": "accept-session", "label": label, "value": "yes-session"})
    options.append({"type": "reject", "label": "No", "value": "no"})
    return options


def _bash_options(command: str) -> list[dict[str, Any]]:
    options = [{"type": "accept-once", "label": "Yes", "value": "yes"}]
    prefix = get_bash_command_prefix(command)
    if prefix:
        options.append({
            "type": "accept-session",
            "scope": "bash-prefix",
            "label": f"Yes, and allow {prefix}:* during this session",
            "value": "yes-session",
            "prefix": prefix,
        })
    options.append({"type": "reject", "label": "No", "value": "no"})
    return options


def _skill_options(skill: str) -> list[dict[str, Any]]:
    return [
        {"type": "accept-once", "label": "Yes", "value": "yes"},
        {"type": "accept-session", "scope": "skill", "label": f"Yes, and allow Skill({skill}) during this session", "value": "yes-session", "skill": skill},
        {"type": "accept-session", "scope": "skill-prefix", "label": f"Yes, and allow Skill({skill}:*) during this session", "value": "yes-prefix-session", "skill": f"{skill}:*"},
        {"type": "reject", "label": "No", "value": "no"},
    ]


def _web_fetch_options(domain: str) -> list[dict[str, Any]]:
    return [
        {"type": "accept-once", "label": "Yes", "value": "yes"},
        {"type": "accept-session", "scope": "web-domain", "label": f"Yes, and allow WebFetch for {domain} during this session", "value": "yes-session", "domain": domain},
        {"type": "reject", "label": "No", "value": "no"},
    ]


def _web_search_options() -> list[dict[str, Any]]:
    return [
        {"type": "accept-once", "label": "Yes", "value": "yes"},
        {"type": "accept-session", "scope": "web-search", "label": "Yes, and allow WebSearch during this session", "value": "yes-session"},
        {"type": "reject", "label": "No", "value": "no"},
    ]
