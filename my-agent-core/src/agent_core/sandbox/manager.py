"""Bash 沙箱管理器。

设计目标对齐 Claude Code：权限模式只决定是否需要用户确认；沙箱是独立的
OS 级执行边界。即使处于 yolo/bypassPermissions，Bash 仍默认进沙箱，除非
单条命令显式请求 dangerously_disable_sandbox 且配置允许 unsandboxed fallback。

当前实现提供 macOS sandbox-exec 的强执行路径，以及 Linux bubblewrap 的保守
best-effort 路径；网络 host 级白名单在开源系统工具里无法稳定表达，因此当
配置了 allowed/denied domains 时默认采取 deny network 的安全退化，并在状态中
暴露 warning，避免静默失效。
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


PlatformName = Literal["darwin", "linux", "unsupported"]


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_SAFE_WRAPPERS = {"time", "timeout", "env", "nice", "nohup", "stdbuf"}
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
_COMPOUND_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_GLOB_CHARS = set("*?[]")


@dataclass(slots=True)
class SandboxDependencyCheck:
    supported: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    platform: PlatformName = "unsupported"


@dataclass(slots=True)
class NetworkSandboxConfig:
    allowed_domains: list[str] = field(default_factory=list)
    denied_domains: list[str] = field(default_factory=list)
    allow_managed_domains_only: bool = False
    allow_unix_sockets: list[str] = field(default_factory=list)
    allow_all_unix_sockets: bool = False
    allow_local_binding: bool = False
    http_proxy_port: int | None = None
    socks_proxy_port: int | None = None


@dataclass(slots=True)
class FilesystemSandboxConfig:
    allow_write: list[str] = field(default_factory=lambda: ["."])
    deny_write: list[str] = field(default_factory=list)
    deny_read: list[str] = field(default_factory=list)
    allow_read: list[str] = field(default_factory=list)
    allow_managed_read_paths_only: bool = False


@dataclass(slots=True)
class SandboxSettings:
    enabled: bool = False
    fail_if_unavailable: bool = False
    enabled_platforms: list[str] | None = None
    auto_allow_bash_if_sandboxed: bool = True
    allow_unsandboxed_commands: bool = True
    excluded_commands: list[str] = field(default_factory=list)
    network: NetworkSandboxConfig = field(default_factory=NetworkSandboxConfig)
    filesystem: FilesystemSandboxConfig = field(default_factory=FilesystemSandboxConfig)
    ignore_violations: dict[str, list[str]] = field(default_factory=dict)
    enable_weaker_nested_sandbox: bool = False
    enable_weaker_network_isolation: bool = False
    ripgrep: dict[str, Any] = field(default_factory=dict)


def _config_path() -> Path:
    return Path(os.getenv("AGENT_SANDBOX_CONFIG") or Path.home() / ".my-agent" / "sandbox.json").expanduser()


def _truthy_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    return None


def _platform_name() -> PlatformName:
    sys = platform.system().lower()
    if sys == "darwin":
        return "darwin"
    if sys == "linux":
        return "linux"
    return "unsupported"


def _merge_dataclass(instance: Any, raw: dict[str, Any]) -> Any:
    for key, value in raw.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def _default_sensitive_write_denies(cwd: Path) -> list[str]:
    home = Path.home()
    return [
        str(home / ".ssh"),
        str(home / ".aws"),
        str(home / ".gnupg"),
        str(home / ".kube"),
        str(home / ".my-agent"),
        str(cwd / ".claude" / "skills"),
        str(cwd / ".claude" / "commands"),
        str(cwd / ".claude" / "agents"),
        str(cwd / ".git" / "hooks"),
    ]


def _resolve_path(pattern: str, cwd: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(pattern))
    path = Path(expanded)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve(strict=False)


def _strip_trailing_glob(path: str) -> str:
    result = path
    for marker in ("/**", "/*"):
        if result.endswith(marker):
            result = result[: -len(marker)]
    return result


def _has_glob(path: str) -> bool:
    return any(ch in path for ch in _GLOB_CHARS)


def _quote_profile_string(value: str) -> str:
    return json.dumps(value)


def _seatbelt_path_filter(kind: str, path: Path) -> str:
    op = "subpath" if path.is_dir() or str(path).endswith(os.sep) else "literal"
    return f'({kind} ({op} {_quote_profile_string(str(path))}))'


class SandboxManager:
    def __init__(self, *, cwd: str | Path | None = None, config_path: str | Path | None = None) -> None:
        self.cwd = Path(cwd or os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()
        self.config_path = Path(config_path).expanduser() if config_path else _config_path()
        self._settings = self._load_settings()

    # ── settings ──────────────────────────────────────────────
    def _load_settings(self) -> SandboxSettings:
        settings = SandboxSettings()
        settings.filesystem.deny_write.extend(_default_sensitive_write_denies(self.cwd))
        if self.config_path.exists():
            try:
                raw = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    _merge_dataclass(settings, _normalize_settings_keys(raw))
            except Exception:
                # 配置损坏时保持安全默认值；Admin/status 会通过 warnings 暴露。
                pass
        self._apply_env_overrides(settings)
        return settings

    def reload(self) -> None:
        self._settings = self._load_settings()

    @property
    def settings(self) -> SandboxSettings:
        return self._settings

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self._settings)

    def set_settings(self, **updates: Any) -> SandboxSettings:
        data = self.settings_dict()
        _deep_update(data, _normalize_settings_keys(updates))
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.reload()
        return self._settings

    def update_raw(self, raw: dict[str, Any]) -> SandboxSettings:
        normalized = _normalize_settings_keys(raw)
        candidate = SandboxSettings()
        _merge_dataclass(candidate, normalized)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(asdict(candidate), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.reload()
        return self._settings

    def _apply_env_overrides(self, settings: SandboxSettings) -> None:
        mapping = {
            "AGENT_SANDBOX_ENABLED": "enabled",
            "AGENT_SANDBOX_FAIL_IF_UNAVAILABLE": "fail_if_unavailable",
            "AGENT_SANDBOX_AUTO_ALLOW_BASH": "auto_allow_bash_if_sandboxed",
            "AGENT_SANDBOX_ALLOW_UNSANDBOXED": "allow_unsandboxed_commands",
        }
        for env_name, attr in mapping.items():
            value = _truthy_env(env_name)
            if value is not None:
                setattr(settings, attr, value)
        if os.getenv("AGENT_SANDBOX_EXCLUDED_COMMANDS"):
            settings.excluded_commands = [p.strip() for p in os.getenv("AGENT_SANDBOX_EXCLUDED_COMMANDS", "").split(",") if p.strip()]
        if os.getenv("AGENT_SANDBOX_ALLOWED_DOMAINS"):
            settings.network.allowed_domains = [p.strip() for p in os.getenv("AGENT_SANDBOX_ALLOWED_DOMAINS", "").split(",") if p.strip()]
        if os.getenv("AGENT_SANDBOX_DENIED_DOMAINS"):
            settings.network.denied_domains = [p.strip() for p in os.getenv("AGENT_SANDBOX_DENIED_DOMAINS", "").split(",") if p.strip()]

    # ── capability/status ─────────────────────────────────────
    def check_dependencies(self) -> SandboxDependencyCheck:
        current = _platform_name()
        errors: list[str] = []
        warnings: list[str] = []
        if current == "unsupported":
            errors.append(f"unsupported platform: {platform.system()}")
        elif current == "darwin":
            if not shutil.which("sandbox-exec"):
                errors.append("sandbox-exec not found")
            if self._settings.network.allowed_domains or self._settings.network.denied_domains:
                warnings.append("macOS sandbox-exec cannot enforce host-level allow/deny; network is denied when domain rules are configured")
        elif current == "linux":
            if not shutil.which("bwrap"):
                errors.append("bubblewrap (bwrap) not found")
            if self._settings.network.allowed_domains or self._settings.network.denied_domains:
                warnings.append("bubblewrap cannot enforce host-level allow/deny without a proxy; network is denied when domain rules are configured")
        if self.config_path.exists():
            try:
                json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                warnings.append(f"sandbox config parse failed: {exc}")
        for path in self.get_linux_glob_pattern_warnings():
            warnings.append(f"glob pattern is not fully supported by OS sandbox: {path}")
        return SandboxDependencyCheck(supported=not errors, errors=errors, warnings=warnings, platform=current)

    def is_supported_platform(self) -> bool:
        return self.check_dependencies().platform in {"darwin", "linux"}

    def is_platform_enabled(self) -> bool:
        allowed = self._settings.enabled_platforms
        return not allowed or _platform_name() in allowed

    def is_sandboxing_enabled(self) -> bool:
        return bool(self._settings.enabled and self.is_platform_enabled() and self.check_dependencies().supported)

    def get_sandbox_unavailable_reason(self) -> str | None:
        if not self._settings.enabled:
            return None
        if not self.is_platform_enabled():
            return f"sandbox.enabled is set but {_platform_name()} is not in enabled_platforms"
        check = self.check_dependencies()
        if check.errors:
            return "; ".join(check.errors)
        return None

    def are_unsandboxed_commands_allowed(self) -> bool:
        return bool(self._settings.allow_unsandboxed_commands)

    def is_auto_allow_bash_if_sandboxed_enabled(self) -> bool:
        return bool(self._settings.auto_allow_bash_if_sandboxed)

    def get_excluded_commands(self) -> list[str]:
        return list(self._settings.excluded_commands)

    def get_fs_read_config(self) -> dict[str, list[str]]:
        return {"deny_read": list(self._settings.filesystem.deny_read), "allow_read": list(self._settings.filesystem.allow_read)}

    def get_fs_write_config(self) -> dict[str, list[str]]:
        return {"allow_write": list(self._settings.filesystem.allow_write), "deny_write": list(self._settings.filesystem.deny_write)}

    def get_network_config(self) -> dict[str, Any]:
        return asdict(self._settings.network)

    def get_linux_glob_pattern_warnings(self) -> list[str]:
        paths = [
            *self._settings.filesystem.allow_write,
            *self._settings.filesystem.deny_write,
            *self._settings.filesystem.deny_read,
            *self._settings.filesystem.allow_read,
        ]
        return [p for p in paths if _has_glob(_strip_trailing_glob(p))]

    # ── command decisions ─────────────────────────────────────
    def should_use_sandbox(self, tool_input: dict[str, Any]) -> bool:
        if not self.is_sandboxing_enabled():
            return False
        dangerous_disable = bool(tool_input.get("dangerously_disable_sandbox") or tool_input.get("dangerouslyDisableSandbox"))
        if dangerous_disable and self.are_unsandboxed_commands_allowed():
            return False
        command = str(tool_input.get("command") or tool_input.get("cmd") or "")
        if not command.strip():
            return False
        return not self.contains_excluded_command(command)

    def contains_excluded_command(self, command: str) -> bool:
        patterns = self._settings.excluded_commands
        if not patterns:
            return False
        subcommands = [part for part in _COMPOUND_SPLIT_RE.split(command) if part.strip()] or [command]
        for subcommand in subcommands:
            candidates = _command_candidates(subcommand)
            for pattern in patterns:
                for candidate in candidates:
                    if _command_matches_pattern(candidate, pattern):
                        return True
        return False

    # ── wrapping ──────────────────────────────────────────────
    def wrap_command(self, command: str, *, cwd: str | Path | None = None, shell: str = "/bin/bash") -> str:
        current = _platform_name()
        run_cwd = Path(cwd or self.cwd).expanduser().resolve()
        if current == "darwin":
            return self._wrap_darwin(command, cwd=run_cwd, shell=shell)
        if current == "linux":
            return self._wrap_linux(command, cwd=run_cwd, shell=shell)
        return command

    def _network_should_be_denied(self) -> bool:
        net = self._settings.network
        return bool(net.allowed_domains or net.denied_domains) and not net.allow_all_unix_sockets

    def _wrap_darwin(self, command: str, *, cwd: Path, shell: str) -> str:
        profile = self._build_seatbelt_profile(cwd)
        return " ".join([
            "sandbox-exec",
            "-p",
            shlex.quote(profile),
            shlex.quote(shell if Path(shell).exists() else "/bin/bash"),
            "-lc",
            shlex.quote(command),
        ])

    def _build_seatbelt_profile(self, cwd: Path) -> str:
        fs = self._settings.filesystem
        lines = ["(version 1)", "(allow default)"]
        lines.append("(deny file-write*)")
        write_paths = [*_default_sensitive_write_denies(cwd), *fs.deny_write]
        allow_paths = [".", tempfile.gettempdir(), *fs.allow_write]
        for raw in allow_paths:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            lines.append(_seatbelt_path_filter("allow file-write*", path))
        for raw in write_paths:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            lines.append(_seatbelt_path_filter("deny file-write*", path))
        for raw in fs.deny_read:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            lines.append(_seatbelt_path_filter("deny file-read*", path))
        for raw in fs.allow_read:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            lines.append(_seatbelt_path_filter("allow file-read*", path))
        if self._network_should_be_denied():
            lines.append("(deny network*)")
            if self._settings.network.allow_local_binding:
                lines.append("(allow network-bind)")
        return "\n".join(lines)

    def _wrap_linux(self, command: str, *, cwd: Path, shell: str) -> str:
        args = ["bwrap", "--die-with-parent", "--proc", "/proc", "--dev", "/dev", "--ro-bind", "/", "/"]
        if self._network_should_be_denied():
            args.append("--unshare-net")
        for raw in [".", tempfile.gettempdir(), *self._settings.filesystem.allow_write]:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            args.extend(["--bind", str(path), str(path)])
        for raw in [*_default_sensitive_write_denies(cwd), *self._settings.filesystem.deny_write, *self._settings.filesystem.deny_read]:
            path = _resolve_path(_strip_trailing_glob(str(raw)), cwd)
            if path.exists():
                args.extend(["--tmpfs", str(path)])
        args.extend(["--chdir", str(cwd), shell if Path(shell).exists() else "/bin/bash", "-lc", command])
        return " ".join(shlex.quote(part) for part in args)

    def cleanup_after_command(self) -> None:
        # 预留与 Claude Code cleanupAfterCommand 对齐；当前 macOS 无需清理。
        return None

    def status(self) -> dict[str, Any]:
        check = self.check_dependencies()
        return {
            "enabled_in_settings": self._settings.enabled,
            "enabled": self.is_sandboxing_enabled(),
            "platform": check.platform,
            "supported": check.supported,
            "errors": check.errors,
            "warnings": check.warnings,
            "unavailable_reason": self.get_sandbox_unavailable_reason(),
            "config_path": str(self.config_path),
            "settings": self.settings_dict(),
        }


def _normalize_settings_keys(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "failIfUnavailable": "fail_if_unavailable",
        "enabledPlatforms": "enabled_platforms",
        "autoAllowBashIfSandboxed": "auto_allow_bash_if_sandboxed",
        "allowUnsandboxedCommands": "allow_unsandboxed_commands",
        "excludedCommands": "excluded_commands",
        "ignoreViolations": "ignore_violations",
        "enableWeakerNestedSandbox": "enable_weaker_nested_sandbox",
        "enableWeakerNetworkIsolation": "enable_weaker_network_isolation",
        "allowedDomains": "allowed_domains",
        "deniedDomains": "denied_domains",
        "allowManagedDomainsOnly": "allow_managed_domains_only",
        "allowUnixSockets": "allow_unix_sockets",
        "allowAllUnixSockets": "allow_all_unix_sockets",
        "allowLocalBinding": "allow_local_binding",
        "httpProxyPort": "http_proxy_port",
        "socksProxyPort": "socks_proxy_port",
        "allowWrite": "allow_write",
        "denyWrite": "deny_write",
        "denyRead": "deny_read",
        "allowRead": "allow_read",
        "allowManagedReadPathsOnly": "allow_managed_read_paths_only",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        new_key = aliases.get(key, key)
        normalized[new_key] = _normalize_settings_keys(value) if isinstance(value, dict) else value
    return normalized


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _command_candidates(command: str) -> list[str]:
    try:
        tokens = shlex.split(command.strip())
    except ValueError:
        tokens = command.strip().split()
    candidates = []
    while tokens and _ENV_ASSIGN_RE.match(tokens[0]):
        tokens = tokens[1:]
    if tokens:
        candidates.append(" ".join(tokens))
        while tokens and tokens[0] in _SAFE_WRAPPERS:
            wrapper = tokens[0]
            tokens = tokens[1:]
            if wrapper == "timeout":
                while tokens and (tokens[0].startswith("-") or re.fullmatch(r"\d+[smhd]?", tokens[0])):
                    tokens = tokens[1:]
            while tokens and _ENV_ASSIGN_RE.match(tokens[0]):
                tokens = tokens[1:]
            if tokens:
                candidates.append(" ".join(tokens))
    return candidates or [command.strip()]


def _command_matches_pattern(command: str, pattern: str) -> bool:
    pat = pattern.strip()
    if not pat:
        return False
    if pat.endswith(":*"):
        prefix = pat[:-2].strip()
        return command == prefix or command.startswith(prefix + " ")
    if any(ch in pat for ch in "*?[]"):
        return re.fullmatch(fnmatch_translate(pat), command) is not None
    return command == pat or command.startswith(pat + " ")


def fnmatch_translate(pattern: str) -> str:
    import fnmatch

    return fnmatch.translate(pattern)


_DEFAULT_MANAGER: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = SandboxManager()
    return _DEFAULT_MANAGER


def should_use_sandbox(tool_input: dict[str, Any]) -> bool:
    return get_sandbox_manager().should_use_sandbox(tool_input)
