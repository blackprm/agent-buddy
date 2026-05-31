"""Hooks 执行引擎 — 匹配、执行、聚合。"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from agent_core.hooks.executors import exec_command_hook, exec_http_hook
from agent_core.hooks.types import (
    AggregatedHookResult,
    HookDefinition,
    HookEvent,
    HookInput,
    HookMatcher,
    HookOutcome,
    HookResult,
    HookType,
)

logger = logging.getLogger(__name__)


class HookEngine:
    """钩子执行引擎。

    用法:
        engine = HookEngine(matchers=[...])
        result = await engine.fire(HookEvent.PreToolUse, tool_name="Write", ...)
        if result.should_block: ...
    """

    def __init__(self, matchers: list[HookMatcher] | None = None) -> None:
        self._matchers: list[HookMatcher] = matchers or []
        # 运行时注册的回调钩子（非持久化，SDK 用）
        self._callbacks: dict[HookEvent, list[Any]] = {}

    # ── 配置管理 ──────────────────────────────────────────

    @property
    def matchers(self) -> list[HookMatcher]:
        return self._matchers

    def set_matchers(self, matchers: list[HookMatcher]) -> None:
        self._matchers = matchers

    def add_matcher(self, matcher: HookMatcher) -> None:
        self._matchers.append(matcher)

    def register_callback(self, event: HookEvent, callback: Any) -> None:
        """注册运行时回调（非持久化）。"""
        self._callbacks.setdefault(event, []).append(callback)

    def get_matching_hooks_for_event(self, event: HookEvent) -> list[Any]:
        """Return persistent hooks/callbacks registered for an event."""
        persistent = [hook for matcher in self._matchers if matcher.event == event for hook in matcher.hooks]
        return [*persistent, *self._callbacks.get(event, [])]

    def clone(self) -> "HookEngine":
        """复制 HookEngine，隔离 once/_executed 等运行时状态。"""
        cloned = HookEngine(matchers=deepcopy(self._matchers))
        cloned._callbacks = {event: list(callbacks) for event, callbacks in self._callbacks.items()}
        return cloned

    # ── 匹配 ─────────────────────────────────────────────

    def get_matching_hooks(self, hook_input: HookInput) -> list[HookDefinition]:
        """获取匹配当前上下文的所有钩子定义。"""
        hooks: list[HookDefinition] = []
        seen_ids: set[str] = set()

        for matcher in self._matchers:
            if matcher.matches(hook_input):
                for hook in matcher.hooks:
                    # 去重（按 name 或 command/url）
                    hook_id = hook.name or hook.command or hook.url
                    if hook_id in seen_ids:
                        continue
                    seen_ids.add(hook_id)

                    # once 检查
                    if hook.once and hook._executed:
                        continue

                    hooks.append(hook)

        return hooks

    # ── 执行 ─────────────────────────────────────────────

    async def fire(
        self,
        event: HookEvent,
        *,
        session_id: str = "",
        cwd: str = "",
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
        tool_use_id: str = "",
        prompt: str = "",
        message_count: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> AggregatedHookResult:
        """触发钩子事件，返回聚合结果。"""
        hook_input = HookInput(
            event=event,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_use_id=tool_use_id,
            prompt=prompt,
            message_count=message_count,
            extra=extra or {},
        )

        # 1. 执行匹配到的持久化钩子
        matching = self.get_matching_hooks(hook_input)
        results: list[HookResult] = []

        for hook_def in matching:
            try:
                result = await self._execute_single(hook_def, hook_input)
                results.append(result)
                hook_def._executed = True
            except Exception as exc:
                logger.exception("Hook execution failed: %s", hook_def.name or hook_def.command)
                results.append(HookResult(
                    hook=hook_def,
                    outcome=HookOutcome.NonBlockingError,
                    stop_reason=str(exc),
                ))

        # 2. 执行运行时回调
        for cb in self._callbacks.get(event, []):
            try:
                cb_result = cb(hook_input)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if isinstance(cb_result, HookResult):
                    results.append(cb_result)
            except Exception as exc:
                logger.exception("Hook callback failed")
                results.append(HookResult(
                    hook=HookDefinition(name="callback"),
                    outcome=HookOutcome.NonBlockingError,
                    stop_reason=str(exc),
                ))

        return AggregatedHookResult(results=results)

    async def _execute_single(
        self, hook: HookDefinition, hook_input: HookInput
    ) -> HookResult:
        """执行单个钩子。"""
        if hook.type == HookType.Command:
            return await exec_command_hook(hook, hook_input)
        elif hook.type == HookType.Http:
            return await exec_http_hook(hook, hook_input)
        else:
            return HookResult(
                hook=hook,
                outcome=HookOutcome.NonBlockingError,
                stop_reason=f"Unknown hook type: {hook.type}",
            )

    # ── 便捷方法 ─────────────────────────────────────────

    async def fire_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        session_id: str = "",
        cwd: str = "",
        tool_use_id: str = "",
    ) -> AggregatedHookResult:
        """触发 PreToolUse 钩子。"""
        return await self.fire(
            HookEvent.PreToolUse,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
        )

    async def fire_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        session_id: str = "",
        cwd: str = "",
        tool_use_id: str = "",
    ) -> AggregatedHookResult:
        """触发 PostToolUse 钩子。"""
        return await self.fire(
            HookEvent.PostToolUse,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
        )

    async def fire_post_tool_use_failure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        session_id: str = "",
        cwd: str = "",
        tool_use_id: str = "",
        error: str = "",
        category: str = "",
        recovery_hint: str = "",
    ) -> AggregatedHookResult:
        """触发 PostToolUseFailure 钩子。"""
        return await self.fire(
            HookEvent.PostToolUseFailure,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            extra={"error": error, "category": category, "recovery_hint": recovery_hint},
        )

    async def fire_permission_denied(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        session_id: str = "",
        cwd: str = "",
        tool_use_id: str = "",
        reason: str = "",
    ) -> AggregatedHookResult:
        """触发 PermissionDenied 钩子，可返回 system_message 提示模型重试。"""
        return await self.fire(
            HookEvent.PermissionDenied,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            extra={"reason": reason},
        )

    async def fire_recovery_suggested(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        session_id: str = "",
        cwd: str = "",
        tool_use_id: str = "",
        category: str = "",
        recovery_hint: str = "",
    ) -> AggregatedHookResult:
        """触发 RecoverySuggested 钩子。"""
        return await self.fire(
            HookEvent.RecoverySuggested,
            session_id=session_id,
            cwd=cwd,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id,
            extra={"category": category, "recovery_hint": recovery_hint},
        )

    async def fire_stop(
        self,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 Stop 钩子。"""
        return await self.fire(
            HookEvent.Stop,
            session_id=session_id,
            cwd=cwd,
        )

    async def fire_session_start(
        self,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 SessionStart 钩子。"""
        return await self.fire(
            HookEvent.SessionStart,
            session_id=session_id,
            cwd=cwd,
        )

    async def fire_user_prompt_submit(
        self,
        prompt: str,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 UserPromptSubmit 钩子。"""
        return await self.fire(
            HookEvent.UserPromptSubmit,
            session_id=session_id,
            cwd=cwd,
            prompt=prompt,
        )

    async def fire_pre_compact(
        self,
        message_count: int,
        *,
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 PreCompact 钩子。"""
        return await self.fire(
            HookEvent.PreCompact,
            session_id=session_id,
            cwd=cwd,
            message_count=message_count,
        )

    async def fire_task_created(
        self,
        *,
        task: dict[str, Any],
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 TaskCreated 钩子。"""
        return await self.fire(
            HookEvent.TaskCreated,
            session_id=session_id,
            cwd=cwd,
            extra={"task": task, "task_id": task.get("id"), "subject": task.get("subject")},
        )

    async def fire_task_completed(
        self,
        *,
        task: dict[str, Any],
        session_id: str = "",
        cwd: str = "",
    ) -> AggregatedHookResult:
        """触发 TaskCompleted 钩子。"""
        return await self.fire(
            HookEvent.TaskCompleted,
            session_id=session_id,
            cwd=cwd,
            extra={"task": task, "task_id": task.get("id"), "subject": task.get("subject")},
        )
