"""Hooks 类型定义 — 借鉴 Claude Code 的 hooks 系统。

核心概念：
- HookEvent: 钩子触发时机（PreToolUse, PostToolUse, Stop 等）
- HookCommand: 钩子定义（command / http 两种类型）
- HookMatcher: 事件匹配规则（正则匹配工具名等）
- HookResult: 钩子执行结果（allow / deny / ask / error）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 钩子事件类型 ──────────────────────────────────────────────


class HookEvent(str, Enum):
    """钩子触发时机。"""

    # 工具相关
    PreToolUse = "PreToolUse"
    PostToolUse = "PostToolUse"
    PostToolUseFailure = "PostToolUseFailure"
    PermissionDenied = "PermissionDenied"
    RecoverySuggested = "RecoverySuggested"

    # 用户交互
    UserPromptSubmit = "UserPromptSubmit"

    # 会话生命周期
    SessionStart = "SessionStart"
    Stop = "Stop"

    # 压缩
    PreCompact = "PreCompact"
    PostCompact = "PostCompact"

    # 任务系统
    TaskCreated = "TaskCreated"
    TaskCompleted = "TaskCompleted"

    # 隔离工作区生命周期
    WorktreeCreate = "WorktreeCreate"
    WorktreeRemove = "WorktreeRemove"


# 支持匹配器的事件（需要 matcher 字段）
_MATCHER_EVENTS = {
    HookEvent.PreToolUse,
    HookEvent.PostToolUse,
    HookEvent.PostToolUseFailure,
    HookEvent.PermissionDenied,
    HookEvent.RecoverySuggested,
}


# ── 钩子输入 ──────────────────────────────────────────────────


@dataclass
class HookInput:
    """钩子执行的输入数据。"""

    event: HookEvent
    session_id: str = ""
    cwd: str = ""

    # 工具相关字段
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""

    # 用户相关字段
    prompt: str = ""

    # 压缩相关
    message_count: int = 0

    # 通用扩展
    extra: dict[str, Any] = field(default_factory=dict)

    def to_env(self) -> dict[str, str]:
        """转换为环境变量（注入到 command hook 进程）。"""
        env: dict[str, str] = {
            "HOOK_EVENT": self.event.value,
            "SESSION_ID": self.session_id,
            "CWD": self.cwd,
        }
        if self.tool_name:
            env["TOOL_NAME"] = self.tool_name
        if self.tool_use_id:
            env["TOOL_USE_ID"] = self.tool_use_id
        if self.prompt:
            env["USER_PROMPT"] = self.prompt
        return env

    def to_json(self) -> dict[str, Any]:
        """转换为 JSON 输入（注入到 http hook / command stdin）。"""
        d: dict[str, Any] = {
            "event": self.event.value,
            "session_id": self.session_id,
            "cwd": self.cwd,
        }
        if self.tool_name:
            d["tool_name"] = self.tool_name
            d["tool_input"] = self.tool_input
            d["tool_use_id"] = self.tool_use_id
        if self.prompt:
            d["prompt"] = self.prompt
        if self.message_count:
            d["message_count"] = self.message_count
        d.update(self.extra)
        return d


# ── 钩子类型 ──────────────────────────────────────────────────


class HookType(str, Enum):
    Command = "command"
    Http = "http"


# ── 钩子定义 ──────────────────────────────────────────────────


@dataclass
class HookDefinition:
    """单个钩子定义。"""

    type: HookType = HookType.Command

    # command 类型
    command: str = ""
    shell: str = "bash"
    timeout: float = 30.0

    # http 类型
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    allowed_env_vars: list[str] = field(default_factory=list)

    # 通用
    name: str = ""  # 显示名称
    if_condition: str = ""  # 条件表达式（暂不实现，预留）
    once: bool = False  # 只执行一次

    # 运行时状态（非持久化）
    _executed: bool = field(default=False, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HookDefinition:
        t = d.get("type", "command")
        return cls(
            type=HookType(t),
            command=d.get("command", ""),
            shell=d.get("shell", "bash"),
            timeout=float(d.get("timeout", 30)),
            url=d.get("url", ""),
            headers=d.get("headers", {}),
            allowed_env_vars=d.get("allowed_env_vars", []),
            name=d.get("name", ""),
            if_condition=d.get("if", ""),
            once=d.get("once", False),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value}
        if self.type == HookType.Command:
            d["command"] = self.command
            if self.shell != "bash":
                d["shell"] = self.shell
            if self.timeout != 30.0:
                d["timeout"] = self.timeout
        elif self.type == HookType.Http:
            d["url"] = self.url
            if self.headers:
                d["headers"] = self.headers
            if self.allowed_env_vars:
                d["allowed_env_vars"] = self.allowed_env_vars
            if self.timeout != 30.0:
                d["timeout"] = self.timeout
        if self.name:
            d["name"] = self.name
        if self.if_condition:
            d["if"] = self.if_condition
        if self.once:
            d["once"] = self.once
        return d


# ── 钩子匹配器 ────────────────────────────────────────────────


@dataclass
class HookMatcher:
    """事件匹配规则。一个 matcher 对应一组 hooks。"""

    event: HookEvent
    matcher: str = ""  # 正则匹配工具名等，空或 * 匹配所有
    hooks: list[HookDefinition] = field(default_factory=list)

    def matches(self, context: HookInput) -> bool:
        """判断是否匹配当前上下文。"""
        if context.event != self.event:
            return False
        if not self.matcher or self.matcher == "*":
            return True
        # 对工具事件匹配工具名
        if self.event in _MATCHER_EVENTS:
            return bool(re.search(self.matcher, context.tool_name))
        return True

    @classmethod
    def from_dict(cls, event: HookEvent, d: dict[str, Any]) -> HookMatcher:
        hooks = [HookDefinition.from_dict(h) for h in d.get("hooks", [])]
        return cls(
            event=event,
            matcher=d.get("matcher", ""),
            hooks=hooks,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"hooks": [h.to_dict() for h in self.hooks]}
        if self.matcher:
            d["matcher"] = self.matcher
        return d


# ── 钩子执行结果 ──────────────────────────────────────────────


class HookOutcome(str, Enum):
    Success = "success"
    Blocking = "blocking"
    NonBlockingError = "non_blocking_error"
    Cancelled = "cancelled"


class PermissionBehavior(str, Enum):
    Passthrough = "passthrough"  # 不影响权限决策
    Allow = "allow"
    Deny = "deny"
    Ask = "ask"


@dataclass
class HookResult:
    """单个钩子执行结果。"""

    hook: HookDefinition
    outcome: HookOutcome = HookOutcome.Success

    # 权限决策（PreToolUse 专用）
    permission_behavior: PermissionBehavior = PermissionBehavior.Passthrough
    permission_reason: str = ""

    # 阻塞消息
    stop_reason: str = ""

    # 系统消息（注入到对话）
    system_message: str = ""

    # 修改工具输入（PreToolUse 专用）
    updated_input: dict[str, Any] | None = None

    # 额外上下文（PostToolUse 专用）
    additional_context: str = ""

    # 监视路径（FileChanged 专用，预留）
    watch_paths: list[str] = field(default_factory=list)

    # 是否阻止继续
    prevent_continuation: bool = False

    # 执行耗时（秒）
    duration: float = 0.0

    # stdout/stderr（调试用）
    stdout: str = ""
    stderr: str = ""


@dataclass
class AggregatedHookResult:
    """聚合多个钩子的执行结果。"""

    results: list[HookResult] = field(default_factory=list)

    @property
    def permission_behavior(self) -> PermissionBehavior:
        """聚合权限决策：deny > ask > allow > passthrough。"""
        behaviors = [r.permission_behavior for r in self.results]
        if PermissionBehavior.Deny in behaviors:
            return PermissionBehavior.Deny
        if PermissionBehavior.Ask in behaviors:
            return PermissionBehavior.Ask
        if PermissionBehavior.Allow in behaviors:
            return PermissionBehavior.Allow
        return PermissionBehavior.Passthrough

    @property
    def permission_reason(self) -> str:
        """返回第一个非空 reason。"""
        for r in self.results:
            if r.permission_reason:
                return r.permission_reason
        return ""

    @property
    def should_block(self) -> bool:
        """是否有阻塞结果。"""
        return any(r.outcome == HookOutcome.Blocking for r in self.results)

    @property
    def should_prevent_continuation(self) -> bool:
        """是否阻止继续。"""
        return any(r.prevent_continuation for r in self.results)

    @property
    def system_messages(self) -> list[str]:
        """所有系统消息。"""
        return [r.system_message for r in self.results if r.system_message]

    @property
    def updated_input(self) -> dict[str, Any] | None:
        """第一个修改工具输入的结果。"""
        for r in self.results:
            if r.updated_input is not None:
                return r.updated_input
        return None

    @property
    def additional_contexts(self) -> list[str]:
        """所有额外上下文。"""
        return [r.additional_context for r in self.results if r.additional_context]

    @property
    def stop_reason(self) -> str:
        """阻塞原因。"""
        for r in self.results:
            if r.outcome == HookOutcome.Blocking and r.stop_reason:
                return r.stop_reason
        return ""
