from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Awaitable

from agent_core.buddy.companion import companion_payload, get_companion
from agent_core.buddy.observer import observe_companion_reaction
from agent_core.context.builder import ContextBuilder
from agent_core.context.compact import AutoCompactor, AutoCompactConfig, ManualCompactResult, manual_compact
from agent_core.core.events import AgentEvent
from agent_core.hooks.engine import HookEngine
from agent_core.hooks.types import HookEvent, PermissionBehavior
from agent_core.memory.session_memory import SessionMemoryManager
from agent_core.model.base import ModelClient, StreamDelta
from agent_core.model.retry import ModelRetryConfig, ModelRetryEvent, categorize_model_error, stream_with_retries
from agent_core.plan_mode import build_plan_implementation_prompt, merge_plan_metadata, restore_plan_state
from agent_core.permissions.policy import PermissionPolicy, PermissionDecision, StaticPermissionPolicy, merge_permission_metadata, restore_permission_state
from agent_core.recovery import ensure_tool_result_pairing, recover_messages_for_resume, recovery_hint_for_tool_result
from agent_core.recovery.tool_errors import repeated_failure_hint
from agent_core.session.store import SessionStore
from agent_core.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    partition_tool_calls,
    tool_concurrency_group,
    tool_max_concurrency,
)
from agent_core.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock


USER_INTERRUPT_MESSAGE = (
    "[User interrupted the previous turn with Ctrl+C / abort. "
    "Treat any in-progress work from that turn as cancelled; do not continue or retry it unless the user explicitly asks.]"
)


def _env_float(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw.strip())
        except ValueError:
            return default
    return default


def _default_model_stream_idle_timeout_seconds() -> float:
    return _env_float(
        (
            "AGENT_MODEL_STREAM_IDLE_TIMEOUT_SECONDS",
            "MODEL_STREAM_IDLE_TIMEOUT_SECONDS",
        ),
        120.0,
    )


@dataclass(slots=True)
class AgentRuntimeConfig:
    max_turns: int = 100
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    skill_store: Any | None = None
    max_output_tokens_recovery_limit: int = 3
    model_stream_idle_timeout_seconds: float = field(default_factory=_default_model_stream_idle_timeout_seconds)
    """模型流式响应两次 delta 之间的最大空闲时间；<=0 表示禁用 idle timeout。"""
    auto_compact_config: AutoCompactConfig | None = None
    """自动压缩配置。None 表示使用默认配置（200K 上下文窗口）。"""
    session_memory_enabled: bool = True
    """是否启用 session memory 后台提取与 system prompt 注入。"""
    model_id: str = ""
    """当前 runtime 绑定的模型 ID，用于 usage 计费归因。"""
    model_provider: str = ""
    """当前 runtime 绑定的模型提供商。"""
    billing_store: Any | None = None
    """可选计费存储。提供 record_usage(session_id, model_id, usage) 方法。"""
    user_id: str = ""
    """当前 runtime 归属用户 ID，用于 session、billing、quota 归因。"""
    org_id: str = ""
    """当前 runtime 归属组织 ID，用于多租户 quota 归因。"""
    quota_store: Any | None = None
    """可选额度存储。提供 check_preflight()/record_usage() 方法。"""
    model_retry_config: ModelRetryConfig = field(default_factory=ModelRetryConfig)
    """模型/API transient failure retry 配置。"""
    cwd: str = field(default_factory=lambda: os.getcwd())
    """Runtime working directory; all relative tool paths resolve from here."""


class AgentRuntime:
    """通用 Agent Loop — 迁移自 Claude Code sourcemap 的 queryLoop。

    核心机制：
    - 流式模型调用（stream），逐 delta 产出事件
    - 用户交互回路：permission "ask" 暂停 loop 等待外部确认
    - 工具并行执行：partitionToolCalls + is_concurrency_safe 分批并发/串行
    - 错误恢复：max_output_tokens recovery、tool error → ToolResultBlock(is_error=True)
    - 中止支持：abort_event 可从外部取消整个 loop
    - Stop hooks：每轮结束后检查是否应继续

    设计模式对应：
    - Strategy：ModelClient、Tool、PermissionPolicy 可替换。
    - Registry：ToolRegistry 负责工具发现与 schema 暴露。
    - Builder：ContextBuilder 负责 system prompt 分层构建。
    - Template Method：run() 固定 loop 骨架，具体模型/工具策略可插拔。
    - Observer/Event Stream：run() 以事件流输出，方便 Web SSE / WebSocket。
    """

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistry,
        context_builder: ContextBuilder | None = None,
        permission_policy: PermissionPolicy | None = None,
        config: AgentRuntimeConfig | None = None,
        session_store: SessionStore | None = None,
        # 外部可注入的交互回调：当 permission 返回 "ask" 时调用。
        # 兼容旧协议返回 "allow" | "deny"，新协议可返回
        # {"decision": "allow"|"deny", "option": {"type": "accept-session"...}}
        ask_callback: Callable[..., Awaitable[str | dict[str, Any]]] | None = None,
        hook_engine: HookEngine | None = None,
        session_memory: SessionMemoryManager | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context_builder = context_builder or ContextBuilder()
        self._permission_policy = permission_policy or StaticPermissionPolicy()
        self._config = config or AgentRuntimeConfig()
        self._session_store = session_store
        self._ask_callback = ask_callback
        self._hook_engine = hook_engine or HookEngine()
        self._auto_compactor = AutoCompactor(self._config.auto_compact_config)
        self._session_memory = session_memory if self._config.session_memory_enabled else None
        # ── 对话历史：runtime 就是 session ──
        self._messages: list[Message] = []
        self._turn_counter: int = 0
        self._tool_failure_counts: dict[str, int] = {}

        # 如果有 session_store，尝试恢复历史
        if self._session_store:
            self._session_store.create_session(
                session_id=self._config.session_id,
                user_id=self._config.user_id,
                org_id=self._config.org_id,
            )
            session_info = self._session_store.get_session(self._config.session_id)
            saved = self._session_store.load_messages(self._config.session_id)
            if saved:
                self._messages, resume_report = recover_messages_for_resume(saved)
                if resume_report.changed:
                    self._session_store.save_messages(self._config.session_id, self._messages, start_turn=0)
                self._turn_counter = len(saved)
            metadata = (session_info or {}).get("metadata") if session_info else {}
            restore_permission_state(self._config.session_id, metadata)
            restore_plan_state(
                self._config.session_id,
                metadata,
                self._messages,
                cwd=self._config.cwd,
            )

    @property
    def messages(self) -> list[Message]:
        """当前对话历史（只读视图）。"""
        return list(self._messages)

    @property
    def session_id(self) -> str:
        return self._config.session_id

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._messages.clear()
        self._auto_compactor.reset_failures()
        self._context_builder.clear_cache()
        self._turn_counter = 0

    def load_history(self, messages: list[Message]) -> None:
        """从外部加载对话历史（用于 SessionStore 恢复）。"""
        self._messages = list(messages)
        self._turn_counter = len(messages)

    def record_user_cancellation(self, *, reason: str = "external_abort", phase: str = "", persist: bool = True) -> bool:
        """Append a user-visible interrupt marker so the next model turn sees Ctrl+C/abort.

        Tool cancellations often happen outside the normal model loop (for example
        a WebSocket Ctrl+C cancelling the background task).  Without an explicit
        message in history, the next model call cannot know the user intentionally
        stopped the previous turn and may continue stale long-running work.
        """
        if self._messages:
            last = self._messages[-1]
            if last.role == "user" and last.metadata.get("user_interrupt"):
                return False
        message = Message.user(USER_INTERRUPT_MESSAGE)
        message.metadata.update({
            "user_interrupt": True,
            "reason": reason,
            "phase": phase,
        })
        self._messages.append(message)
        self._context_builder.clear_cache()
        if persist:
            self._persist()
        return True

    async def compact_history(self, custom_instructions: str | None = None) -> ManualCompactResult:
        """手动压缩当前会话历史，并把 compact summary 写回 session。"""
        if not self._messages:
            raise ValueError("Not enough messages to compact.")

        system = await self._build_system_prompt()

        result = await manual_compact(
            self._messages,
            self._model,
            system=system,
            custom_instructions=custom_instructions,
            stream_idle_timeout_seconds=self._config.model_stream_idle_timeout_seconds,
        )
        self._messages = result.messages
        self._turn_counter = len(self._messages)
        self._auto_compactor.reset_failures()
        self._context_builder.clear_cache()
        self._persist()
        return result

    def _persist(self) -> None:
        """将当前对话历史持久化到 SessionStore。"""
        if self._session_store:
            self._session_store.save_messages(
                self._config.session_id,
                self._messages,
                start_turn=0,
            )
            session_info = self._session_store.get_session(self._config.session_id)
            metadata = dict((session_info or {}).get("metadata") or {})
            metadata = merge_permission_metadata(metadata, self._config.session_id)
            self._session_store.update_session_metadata(
                self._config.session_id,
                merge_plan_metadata(metadata, self._config.session_id, cwd=self._config.cwd),
            )

    async def run(
        self,
        user_input: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """运行 agent loop，以事件流形式产出每个阶段。

        对话历史由 self._messages 持有，每次 run() 追加用户输入后开始 loop。
        调用方可通过 runtime.messages 读取完整历史，通过 load_history() 恢复。

        参数:
            user_input: 用户输入文本
            abort_event: 外部可设置的 asyncio.Event，设置后 loop 会在安全点中止
        """
        attachments = [dict(item) for item in (attachments or []) if isinstance(item, dict)]
        user_message_text = user_input
        if attachments:
            lines = [user_input.rstrip(), "", "[User uploaded media attachments]"]
            for idx, item in enumerate(attachments, 1):
                attachment_id = str(item.get("id") or item.get("attachment_id") or "").strip()
                filename = str(item.get("filename") or "media")
                content_type = str(item.get("content_type") or "media")
                size_bytes = item.get("size_bytes")
                lines.append(
                    f"{idx}. attachment_id={attachment_id} filename={filename} content_type={content_type} size_bytes={size_bytes}"
                )
            lines.append(
                "Use the UnderstandImage tool with attachment_id for one item or attachment_ids for multiple/mixed images and videos when you need to inspect them."
            )
            user_message_text = "\n".join(lines).strip()
        user_message = Message.user(user_message_text)
        user_message.metadata["original_user_input"] = user_input
        if attachments:
            user_message.metadata["attachments"] = attachments
        self._messages.append(user_message)
        # Persist the user turn immediately so uploaded images/videos remain
        # visible in history replay even if the browser reconnects while the
        # agent is still running.  The full conversation is persisted again at
        # normal loop checkpoints/completion.
        self._persist()
        messages = self._messages
        system = await self._build_system_prompt()
        yield AgentEvent("loop_started", {"session_id": self._config.session_id, "cwd": self._config.cwd, "workspace": self._config.metadata.get("workspace")})

        # ── SessionStart hook ──
        if self._turn_counter == 0:
            session_result = await self._hook_engine.fire_session_start(
                session_id=self._config.session_id,
                cwd=self._config.cwd,
            )
            for msg in session_result.system_messages:
                yield AgentEvent("hook_message", {"message": msg})

        # ── UserPromptSubmit hook ──
        prompt_result = await self._hook_engine.fire_user_prompt_submit(
            user_input,
            session_id=self._config.session_id,
            cwd=self._config.cwd,
        )
        if prompt_result.should_block:
            yield AgentEvent("loop_failed", {"error": prompt_result.stop_reason})
            return
        for msg in prompt_result.system_messages:
            yield AgentEvent("hook_message", {"message": msg})

        max_output_tokens_recovery_count = 0

        try:
            for turn in range(1, self._config.max_turns + 1):
                # ── 检查中止 ──
                if abort_event and abort_event.is_set():
                    self.record_user_cancellation(reason="external_abort", phase="before_turn")
                    yield AgentEvent("loop_aborted", {"turn": turn, "reason": "external_abort"})
                    self._persist()
                    return

                yield AgentEvent("turn_started", {"turn": turn})

                quota_decision = self._check_quota_preflight()
                if quota_decision is not None and not quota_decision.get("allowed", True):
                    yield AgentEvent("quota_exceeded", quota_decision)
                    yield AgentEvent("loop_failed", {"error": quota_decision.get("reason") or "quota exceeded"})
                    self._persist()
                    return
                if quota_decision is not None:
                    yield AgentEvent("quota_checked", quota_decision)

                # ── 1. 流式调用模型 ──
                yield AgentEvent("model_started", {"turn": turn})
                # Rebuild every turn so uncached dynamic sections (notably conditional
                # skills activated by previous tool calls) are visible immediately.
                system = await self._build_system_prompt()
                repaired_messages, repair_report = ensure_tool_result_pairing(messages)
                if repair_report.repaired:
                    self._messages = repaired_messages
                    messages = self._messages
                    yield AgentEvent("conversation_repaired", {"turn": turn, **repair_report.to_dict()})
                content_blocks: list[TextBlock | ThinkingBlock | ToolUseBlock] = []
                tool_uses: list[ToolUseBlock] = []
                stop_reason: str | None = None
                usage: dict[str, Any] = {}

                # 流式收集状态
                current_text = ""
                current_thinking = ""
                current_tool_id = ""
                current_tool_name = ""
                current_tool_input_json = ""

                stream = stream_with_retries(
                    self._model,
                    system=system,
                    messages=messages,
                    tools=self._tools.schemas(),
                    metadata=self._config.metadata,
                    config=self._config.model_retry_config,
                )
                try:
                    async for delta in self._stream_with_idle_timeout(stream, turn=turn):
                        if isinstance(delta, ModelRetryEvent):
                            content_blocks.clear()
                            tool_uses.clear()
                            current_text = ""
                            current_thinking = ""
                            current_tool_id = ""
                            current_tool_name = ""
                            current_tool_input_json = ""
                            yield AgentEvent("api_retry", {"turn": turn, **delta.to_dict()})
                            continue
                        # 检查中止（在流式过程中也可以中断）
                        if abort_event and abort_event.is_set():
                            self.record_user_cancellation(reason="external_abort", phase="streaming")
                            yield AgentEvent("loop_aborted", {"turn": turn, "reason": "external_abort", "phase": "streaming"})
                            self._persist()
                            return

                        if delta.type == "text_delta":
                            current_text += delta.text
                            yield AgentEvent("text_delta", {"turn": turn, "text": delta.text})
                        elif delta.type == "thinking_delta":
                            current_thinking += delta.text
                            yield AgentEvent("thinking_delta", {"turn": turn, "text": delta.text})
                        elif delta.type == "tool_use_start":
                            # flush 之前的 text/thinking/tool_use block
                            if current_thinking:
                                content_blocks.append(ThinkingBlock(thinking=current_thinking))
                                current_thinking = ""
                            if current_text:
                                content_blocks.append(TextBlock(text=current_text))
                                current_text = ""
                            if current_tool_id:
                                try:
                                    tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                except json.JSONDecodeError:
                                    tool_input = {}
                                content_blocks.append(ToolUseBlock(id=current_tool_id, name=current_tool_name, input=tool_input))
                            current_tool_id = delta.tool_use_id
                            current_tool_name = delta.tool_use_name
                            current_tool_input_json = ""
                        elif delta.type == "tool_use_delta":
                            current_tool_input_json += delta.tool_use_input_delta
                        elif delta.type == "stop":
                            # flush 最后的 text/thinking block
                            if current_thinking:
                                content_blocks.append(ThinkingBlock(thinking=current_thinking))
                                current_thinking = ""
                            if current_text:
                                content_blocks.append(TextBlock(text=current_text))
                                current_text = ""
                            # flush 最后的 tool_use block
                            if current_tool_id:
                                try:
                                    tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                except json.JSONDecodeError:
                                    tool_input = {}
                                tool_block = ToolUseBlock(id=current_tool_id, name=current_tool_name, input=tool_input)
                                content_blocks.append(tool_block)
                                current_tool_id = ""
                                current_tool_name = ""
                                current_tool_input_json = ""
                            stop_reason = delta.stop_reason
                            usage = delta.usage
                except Exception as exc:
                    category, _ = categorize_model_error(exc)
                    yield AgentEvent("model_error", {"turn": turn, "category": category, "error": str(exc)})
                    if category == "prompt_too_long":
                        should_compact = True
                        yield AgentEvent("context_compacting", {"turn": turn, "message_count": len(messages), "reason": "prompt_too_long"})
                        compacted, did_compact = await self._auto_compactor.compact_if_needed(messages, self._model, system=system, force=True)
                        if did_compact:
                            self._messages = compacted
                            messages = self._messages
                            yield AgentEvent("context_compacted", {"turn": turn, "message_count": len(messages), "reason": "prompt_too_long"})
                            continue
                        yield AgentEvent("context_compact_failed", {"turn": turn, "message_count": len(messages), "reason": "prompt_too_long"})
                    raise

                # ── 2. 组装 assistant message ──
                assistant_message = Message(role="assistant", content=content_blocks)
                messages.append(assistant_message)
                billing_summary = self._record_billing_usage(usage)
                quota_status = self._record_quota_usage(billing_summary, usage)
                completed_payload: dict[str, Any] = {
                    "turn": turn,
                    "stop_reason": stop_reason,
                    "usage": usage,
                    "model_id": self._config.model_id,
                    "model_provider": self._config.model_provider,
                    "session_id": self._config.session_id,
                }
                if billing_summary is not None:
                    completed_payload["billing"] = billing_summary
                if quota_status is not None:
                    completed_payload["quota"] = quota_status
                yield AgentEvent("model_completed", completed_payload)

                # ── 3. 提取 text 和 tool_use ──
                for block in content_blocks:
                    if isinstance(block, TextBlock) and block.text:
                        yield AgentEvent("assistant_text", {"turn": turn, "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tool_uses.append(block)

                if not tool_uses:
                    # ── max_output_tokens recovery ──
                    if stop_reason == "max_output_tokens":
                        if max_output_tokens_recovery_count < self._config.max_output_tokens_recovery_limit:
                            max_output_tokens_recovery_count += 1
                            recovery_msg = Message.user(
                                "Output token limit hit. Resume directly — pick up mid-thought. "
                                "Break remaining work into smaller pieces."
                            )
                            messages.append(recovery_msg)
                            yield AgentEvent("turn_started", {"turn": turn, "reason": "max_output_tokens_recovery", "attempt": max_output_tokens_recovery_count})
                            continue

                    # ── Stop hook ──
                    stop_result = await self._hook_engine.fire_stop(
                        session_id=self._config.session_id,
                        cwd=self._config.cwd,
                    )
                    if stop_result.should_block:
                        # hook 阻止结束，注入消息让模型继续
                        block_msg = stop_result.stop_reason or "Stop hook requested continuation"
                        messages.append(Message.user(f"[hook] {block_msg}"))
                        yield AgentEvent("hook_message", {"message": f"Stop hook blocked: {block_msg}"})
                        continue
                    for msg in stop_result.system_messages:
                        yield AgentEvent("hook_message", {"message": msg})

                    companion_event = self._companion_reaction_event()
                    if companion_event is not None:
                        yield companion_event

                    yield AgentEvent("loop_completed", {"turns": turn, "stop_reason": stop_reason})
                    self._persist()
                    await self._maybe_extract_session_memory(messages)
                    return

                # ── 4. 权限检查 + 工具执行（partitionToolCalls）──
                tool_result_blocks: list[ToolResultBlock] = []
                batches = partition_tool_calls(tool_uses, self._tools)
                plan_handoff: dict[str, Any] | None = None

                for batch in batches:
                    # 检查中止
                    if abort_event and abort_event.is_set():
                        # 为所有未执行的 tool_use 生成中断结果
                        for tu in batch.blocks:
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content="Interrupted by user", is_error=True)
                            )
                        yield AgentEvent("loop_aborted", {"turn": turn, "reason": "external_abort", "phase": "tool_execution"})
                        # 仍然把 tool_result 追加到 messages，保持 API 兼容
                        messages.append(Message(role="user", content=tool_result_blocks))
                        self.record_user_cancellation(reason="external_abort", phase="tool_execution")
                        self._persist()
                        return

                    # 权限检查
                    allowed_uses: list[ToolUseBlock] = []
                    for tu in batch.blocks:
                        tool = self._tools.get(tu.name)
                        if tool is None:
                            failure = recovery_hint_for_tool_result(tu.name, f"Unknown tool: {tu.name}")
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=f"Unknown tool: {tu.name}", is_error=True)
                            )
                            yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": "unknown", "reason": f"Unknown tool: {tu.name}", **failure.to_metadata()})
                            yield AgentEvent("recovery_suggested", {"tool_use_id": tu.id, "tool": tu.name, "message": failure.message, **failure.to_metadata()})
                            continue

                        # ── PreToolUse hook ──
                        pre_hook = await self._hook_engine.fire_pre_tool_use(
                            tu.name, tu.input,
                            session_id=self._config.session_id,
                            cwd=self._config.cwd,
                            tool_use_id=tu.id,
                        )
                        if pre_hook.should_block:
                            hook_reason = pre_hook.stop_reason or "Blocked by hook"
                            failure = recovery_hint_for_tool_result(tu.name, f"Hook blocked: {hook_reason}")
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=f"Hook blocked: {hook_reason}", is_error=True)
                            )
                            yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": "hook_block", "reason": hook_reason, **failure.to_metadata()})
                            async for evt in self._fire_permission_denied_and_recovery(tu, f"Hook blocked: {hook_reason}", failure):
                                yield evt
                            continue
                        # hook 可以修改工具输入
                        if pre_hook.updated_input is not None:
                            tu = ToolUseBlock(id=tu.id, name=tu.name, input=pre_hook.updated_input)
                        for msg in pre_hook.system_messages:
                            yield AgentEvent("hook_message", {"message": msg})

                        # ── 权限策略检查 ──
                        decision = await self._permission_policy.check(tool=tool, tool_input=tu.input)

                        # hook 权限决策与策略合并：hook deny 优先，hook allow 不能绕过策略 deny/ask
                        hook_perm = pre_hook.permission_behavior
                        if hook_perm == PermissionBehavior.Deny:
                            failure = recovery_hint_for_tool_result(tu.name, f"Hook denied: {pre_hook.permission_reason}")
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=f"Hook denied: {pre_hook.permission_reason}", is_error=True)
                            )
                            yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": "hook_deny", "reason": pre_hook.permission_reason, **failure.to_metadata()})
                            async for evt in self._fire_permission_denied_and_recovery(tu, f"Hook denied: {pre_hook.permission_reason}", failure):
                                yield evt
                            continue
                        if hook_perm == PermissionBehavior.Allow and decision.status == "deny":
                            # hook allow 不能绕过策略 deny
                            pass  # 继续走策略 deny
                        elif hook_perm == PermissionBehavior.Allow and decision.status == "ask":
                            # hook allow 不能绕过策略 ask；仍然要求用户确认
                            pass
                        elif hook_perm == PermissionBehavior.Ask:
                            # hook ask → 强制弹出权限对话框
                            decision = PermissionDecision(status="ask", reason=pre_hook.permission_reason or decision.reason)

                        if decision.status == "allow":
                            allowed_uses.append(tu)
                        elif decision.status == "ask":
                            # ── 用户交互回路 ──
                            yield AgentEvent("permission_request", {
                                "tool_use_id": tu.id,
                                "tool": tu.name,
                                "input": tu.input,
                                "reason": decision.reason,
                                "options": decision.options,
                                "metadata": decision.metadata,
                            })
                            if self._ask_callback:
                                try:
                                    user_response = await self._ask_callback(tu.name, tu.input, decision)
                                except TypeError:
                                    # Backward compatibility for older channels that only accept
                                    # (tool_name, tool_input) and return "allow" | "deny".
                                    user_response = await self._ask_callback(tu.name, tu.input)
                                user_decision = user_response.get("decision") if isinstance(user_response, dict) else user_response
                                user_option = user_response.get("option") if isinstance(user_response, dict) else None
                                if user_decision == "allow":
                                    self._permission_policy.record_user_decision(tool=tool, tool_input=tu.input, option=user_option)
                                    allowed_uses.append(tu)
                                else:
                                    failure = recovery_hint_for_tool_result(tu.name, f"User denied: {decision.reason}")
                                    tool_result_blocks.append(
                                        ToolResultBlock(tool_use_id=tu.id, content=f"User denied: {decision.reason}", is_error=True)
                                    )
                                    yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": "user_deny", "reason": decision.reason, **failure.to_metadata()})
                                    async for evt in self._fire_permission_denied_and_recovery(tu, f"User denied: {decision.reason}", failure):
                                        yield evt
                            else:
                                # 没有 ask_callback 时默认拒绝
                                failure = recovery_hint_for_tool_result(tu.name, f"Permission required but no callback: {decision.reason}")
                                tool_result_blocks.append(
                                    ToolResultBlock(tool_use_id=tu.id, content=f"Permission required but no callback: {decision.reason}", is_error=True)
                                )
                                yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": "ask_no_callback", "reason": decision.reason, **failure.to_metadata()})
                                async for evt in self._fire_permission_denied_and_recovery(tu, f"Permission required but no callback: {decision.reason}", failure):
                                    yield evt
                        else:  # deny
                            failure = recovery_hint_for_tool_result(tu.name, decision.reason)
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=decision.reason, is_error=True)
                            )
                            yield AgentEvent("tool_denied", {"tool_use_id": tu.id, "tool": tu.name, "status": decision.status, "reason": decision.reason, **failure.to_metadata()})
                            async for evt in self._fire_permission_denied_and_recovery(tu, decision.reason, failure):
                                yield evt

                    if not allowed_uses:
                        continue

                    # 执行工具
                    tool_context = ToolContext(
                        session_id=self._config.session_id,
                        messages=messages,
                        metadata=self._config.metadata,
                        cwd=self._config.cwd,
                        ask_callback=self._ask_callback,
                        hook_engine=self._hook_engine,
                    )

                    if batch.is_concurrency_safe:
                        # ── 并发执行 ──
                        completed_results: dict[str, ToolResult] = {}
                        async for item in self._execute_tools_concurrently_with_progress(allowed_uses, tool_context):
                            if isinstance(item, AgentEvent):
                                yield item
                            else:
                                tu, result = item
                                completed_results[tu.id] = result
                                if result.is_error:
                                    async for evt in self._process_tool_failure(tu, result):
                                        yield evt
                                yield AgentEvent("tool_completed", {"tool_use_id": tu.id, "tool": tu.name, "is_error": result.is_error, "result": result.content, "metadata": result.metadata, "concurrent": True})
                                if not result.is_error and isinstance(result.metadata.get("planImplementation"), dict):
                                    plan_handoff = dict(result.metadata["planImplementation"])
                                self._activate_conditional_skills_for_tool(tu)
                                # PostToolUse hook
                                async for evt in self._fire_post_tool_use(tu, result.content):
                                    yield evt

                        for tu in allowed_uses:
                            result = completed_results.get(tu.id)
                            if result is None:
                                result = ToolResult(content=f"Tool {tu.name} produced no result", is_error=True)
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=result.content, is_error=result.is_error, metadata=result.metadata)
                            )
                    else:
                        # ── 串行执行 ──
                        for tu in allowed_uses:
                            yield AgentEvent("tool_started", {"tool_use_id": tu.id, "tool": tu.name, "input": tu.input, "concurrent": False})
                            result: ToolResult | None = None
                            async for item in self._execute_single_tool_with_progress(tu, tool_context):
                                if isinstance(item, AgentEvent):
                                    yield item
                                else:
                                    result = item
                            if result is None:
                                result = ToolResult(content=f"Tool {tu.name} produced no result", is_error=True)
                            if result.is_error:
                                async for evt in self._process_tool_failure(tu, result):
                                    yield evt
                            yield AgentEvent("tool_completed", {"tool_use_id": tu.id, "tool": tu.name, "is_error": result.is_error, "result": result.content, "metadata": result.metadata, "concurrent": False})
                            if not result.is_error and isinstance(result.metadata.get("planImplementation"), dict):
                                plan_handoff = dict(result.metadata["planImplementation"])
                            self._activate_conditional_skills_for_tool(tu)
                            tool_result_blocks.append(
                                ToolResultBlock(tool_use_id=tu.id, content=result.content, is_error=result.is_error, metadata=result.metadata)
                            )
                            # PostToolUse hook
                            async for evt in self._fire_post_tool_use(tu, result.content):
                                yield evt

                # ── 5. 追加 tool_result 到 messages，进入下一轮 ──
                messages.append(Message(role="user", content=tool_result_blocks))

                if plan_handoff:
                    plan = str(plan_handoff.get("plan") or "")
                    clear_context = bool(plan_handoff.get("clearContext"))
                    implementation_prompt = build_plan_implementation_prompt(
                        plan,
                        plan_file_path=plan_handoff.get("planFilePath"),
                        clear_context=clear_context,
                    )
                    implementation_message = Message.user(implementation_prompt)
                    implementation_message.metadata.update({
                        "plan_implementation": True,
                        "plan_file_path": plan_handoff.get("planFilePath"),
                        "clear_context": clear_context,
                    })
                    if clear_context:
                        self._messages = [implementation_message]
                        messages = self._messages
                        self._turn_counter = 0
                        self._auto_compactor.reset_failures()
                    else:
                        messages.append(implementation_message)
                    self._context_builder.clear_cache()
                    yield AgentEvent("plan_implementation_started", {
                        "turn": turn,
                        "clearContext": clear_context,
                        "mode": plan_handoff.get("mode"),
                        "planFilePath": plan_handoff.get("planFilePath"),
                    })
                    self._persist()
                    continue

                # ── 6. 自动压缩检查 ──
                if usage.get("prompt_tokens"):
                    self._auto_compactor.update_exact_count(usage["prompt_tokens"], len(messages) - 1)
                should_compact = self._auto_compactor.should_compact(messages)
                if should_compact:
                    yield AgentEvent("context_compacting", {
                        "turn": turn,
                        "message_count": len(messages),
                    })
                compacted, did_compact = await self._auto_compactor.compact_if_needed(
                    messages, self._model, system=system,
                )
                if did_compact:
                    # compact 可能返回新列表，需要同步回 self._messages
                    self._messages = compacted
                    messages = self._messages
                    yield AgentEvent("context_compacted", {
                        "turn": turn,
                        "message_count": len(messages),
                    })
                elif should_compact:
                    yield AgentEvent("context_compact_failed", {
                        "turn": turn,
                        "message_count": len(messages),
                    })

            yield AgentEvent("loop_failed", {"error": f"max_turns exceeded: {self._config.max_turns}"})
            self._persist()

        except asyncio.CancelledError:
            self.record_user_cancellation(reason="cancelled", phase="task_cancelled")
            yield AgentEvent("loop_aborted", {"reason": "cancelled"})
            self._persist()
        except TimeoutError as exc:
            yield AgentEvent("loop_failed", {"error": str(exc)})
            self._persist()
        except Exception as exc:  # noqa: BLE001 - runtime 边界必须把异常转为事件
            yield AgentEvent("loop_failed", {"error": repr(exc)})
            self._persist()

    async def _execute_single_tool_with_progress(self, tu: ToolUseBlock, base_context: ToolContext) -> AsyncIterator[AgentEvent | ToolResult]:
        """Execute one serial tool while forwarding tool-emitted progress events.

        Tools may call ``context.event_callback(AgentEvent(...))``.  The agent loop
        drains those events while the tool is still running, matching Claude Code's
        long-running Bash progress behavior without changing the public Tool API.
        """
        tool = self._tools.get(tu.name)
        if tool is None:
            yield ToolResult(content=f"Unknown tool: {tu.name}", is_error=True)
            return

        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        async def emit(event: AgentEvent) -> None:
            await queue.put(event)

        context = ToolContext(
            session_id=base_context.session_id,
            messages=base_context.messages,
            metadata={**base_context.metadata, "tool_use_id": tu.id, "tool_name": tu.name},
            cwd=base_context.cwd,
            ask_callback=base_context.ask_callback,
            hook_engine=base_context.hook_engine,
            event_callback=emit,
        )

        async def run_tool() -> ToolResult:
            try:
                return await tool.call(tu.input, context)
            except Exception as exc:  # noqa: BLE001 - tool boundary converts to recoverable result
                from agent_core.recovery.tool_errors import format_tool_exception
                failure = format_tool_exception(tu, exc)
                return ToolResult(content=failure.message, is_error=True, metadata=failure.to_metadata())

        task = asyncio.create_task(run_tool(), name=f"tool-{tu.name}-{tu.id}")
        pending_get: asyncio.Task | None = None
        try:
            while True:
                if pending_get is None:
                    pending_get = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({task, pending_get}, return_when=asyncio.FIRST_COMPLETED)
                if pending_get in done:
                    event = pending_get.result()
                    pending_get = None
                    yield event
                    continue
                if task in done:
                    if pending_get and not pending_get.done():
                        pending_get.cancel()
                    while not queue.empty():
                        yield queue.get_nowait()
                    yield task.result()
                    return
        finally:
            if pending_get and not pending_get.done():
                pending_get.cancel()
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _execute_tools_concurrently_with_progress(self, tool_uses: list[ToolUseBlock], base_context: ToolContext) -> AsyncIterator[AgentEvent | tuple[ToolUseBlock, ToolResult]]:
        """Execute concurrency-safe tools in parallel while forwarding per-tool progress.

        Each tool receives its own ToolContext metadata containing the matching
        tool_use_id.  This is especially important for long-running tools such as
        GenerateVideo: multiple video tasks can now run at the same time while
        heartbeat/progress events are still routed to the correct UI card.
        """
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        pending = list(tool_uses)
        tasks: dict[asyncio.Task[tuple[ToolUseBlock, ToolResult]], tuple[ToolUseBlock, str]] = {}
        active_by_group: dict[str, int] = {}

        def _context_for(tu: ToolUseBlock) -> ToolContext:
            async def emit(event: AgentEvent) -> None:
                data = dict(event.data)
                data.setdefault("tool_use_id", tu.id)
                data.setdefault("tool", tu.name)
                await queue.put(AgentEvent(event.type, data))

            return ToolContext(
                session_id=base_context.session_id,
                messages=base_context.messages,
                metadata={**base_context.metadata, "tool_use_id": tu.id, "tool_name": tu.name},
                cwd=base_context.cwd,
                ask_callback=base_context.ask_callback,
                hook_engine=base_context.hook_engine,
                event_callback=emit,
            )

        async def run_tool(tu: ToolUseBlock) -> tuple[ToolUseBlock, ToolResult]:
            tool = self._tools.get(tu.name)
            if tool is None:
                return tu, ToolResult(content=f"Unknown tool: {tu.name}", is_error=True)
            try:
                return tu, await tool.call(tu.input, _context_for(tu))
            except Exception as exc:  # noqa: BLE001 - tool boundary converts to recoverable result
                from agent_core.recovery.tool_errors import format_tool_exception
                failure = format_tool_exception(tu, exc)
                return tu, ToolResult(content=failure.message, is_error=True, metadata=failure.to_metadata())

        def start_ready_tools() -> list[AgentEvent]:
            started_events: list[AgentEvent] = []
            idx = 0
            while idx < len(pending):
                tu = pending[idx]
                tool = self._tools.get(tu.name)
                group = tool_concurrency_group(tool, tu.name)
                limit = tool_max_concurrency(tool)
                if limit is not None and active_by_group.get(group, 0) >= limit:
                    idx += 1
                    continue

                pending.pop(idx)
                active_by_group[group] = active_by_group.get(group, 0) + 1
                task = asyncio.create_task(run_tool(tu), name=f"tool-{tu.name}-{tu.id}")
                tasks[task] = (tu, group)
                started_events.append(AgentEvent("tool_started", {"tool_use_id": tu.id, "tool": tu.name, "input": tu.input, "concurrent": True}))
            return started_events

        pending_get: asyncio.Task[AgentEvent] | None = None
        try:
            for event in start_ready_tools():
                yield event
            while tasks:
                if pending_get is None:
                    pending_get = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait([*tasks.keys(), pending_get], return_when=asyncio.FIRST_COMPLETED)

                if pending_get in done:
                    yield pending_get.result()
                    done.remove(pending_get)
                    pending_get = None

                for task in [task for task in done if task in tasks]:
                    tu, result = task.result()
                    _, group = tasks[task]
                    tasks.pop(task, None)
                    active_by_group[group] = max(0, active_by_group.get(group, 1) - 1)
                    yield tu, result
                    for event in start_ready_tools():
                        yield event

            if pending_get and not pending_get.done():
                pending_get.cancel()
                pending_get = None
            while not queue.empty():
                yield queue.get_nowait()
        finally:
            if pending_get and not pending_get.done():
                pending_get.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _build_system_prompt(self):
        system = await self._context_builder.build()
        if self._session_memory:
            memory_section = await self._session_memory.context_section()
            if memory_section:
                system.append(memory_section)
        return system

    def _record_billing_usage(self, usage: dict[str, Any]) -> dict[str, Any] | None:
        store = self._config.billing_store
        model_id = self._config.model_id
        if not store or not model_id or not usage:
            return None
        try:
            return store.record_usage(
                session_id=self._config.session_id,
                model_id=model_id,
                usage=usage,
                user_id=self._config.user_id,
                org_id=self._config.org_id,
            )
        except Exception:
            # Billing must never break the agent loop.  The raw usage still
            # remains visible in the model_completed event for troubleshooting.
            return None

    def _check_quota_preflight(self) -> dict[str, Any] | None:
        store = self._config.quota_store
        if not store or not self._config.user_id:
            return None
        try:
            decision = store.check_preflight(user_id=self._config.user_id, org_id=self._config.org_id)
            return decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        except Exception as exc:
            # Quota is a server-side guardrail; surface failures rather than silently
            # bypassing limits, but keep the loop recoverable as an event.
            return {"allowed": False, "reason": f"quota preflight failed: {exc!r}"}

    def _record_quota_usage(self, billing_summary: dict[str, Any] | None, usage: dict[str, Any]) -> dict[str, Any] | None:
        store = self._config.quota_store
        if not store or not self._config.user_id:
            return None
        try:
            usage = usage or {}
            total_tokens = int(usage.get("total_tokens") or 0)
            if not total_tokens:
                total_tokens = sum(
                    int(usage.get(key) or 0)
                    for key in (
                        "prompt_tokens",
                        "input_tokens",
                        "completion_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
            total_cost = self._estimate_usage_cost(usage)
            return store.record_usage(
                user_id=self._config.user_id,
                org_id=self._config.org_id,
                total_tokens=total_tokens,
                total_cost=total_cost,
            )
        except Exception:
            return None

    def _estimate_usage_cost(self, usage: dict[str, Any]) -> float:
        billing_store = self._config.billing_store
        model_id = self._config.model_id
        if not billing_store or not model_id:
            return 0.0
        try:
            pricing = billing_store.get_model(model_id) or {}
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            return (
                prompt * float(pricing.get("input_per_million") or 0)
                + completion * float(pricing.get("output_per_million") or 0)
                + cache_read * float(pricing.get("cache_read_per_million") or 0)
                + cache_creation * float(pricing.get("cache_write_per_million") or 0)
            ) / 1_000_000
        except Exception:
            return 0.0

    def _companion_reaction_event(self) -> AgentEvent | None:
        if not self._config.user_id:
            return None
        try:
            user_context = {
                "user": {
                    "id": self._config.user_id,
                    "account_uuid": self._config.metadata.get("account_uuid") or self._config.user_id,
                },
                "organization": {
                    "id": self._config.org_id,
                    "organization_uuid": self._config.metadata.get("organization_uuid") or self._config.org_id,
                },
            }
            companion = get_companion(user_context, create=False)
            reaction = observe_companion_reaction(self._messages, companion)
            if not reaction:
                return None
            return AgentEvent("companion_reaction", {"text": reaction, "companion": companion_payload(companion)})
        except Exception:
            return None

    def _activate_conditional_skills_for_tool(self, tu: ToolUseBlock) -> None:
        store = self._config.skill_store
        if not store or not hasattr(store, "activate_for_paths"):
            return
        paths: list[str] = []
        for key in ("path", "file_path", "filepath", "filename"):
            value = tu.input.get(key)
            if isinstance(value, str) and value:
                paths.append(value)
        if paths:
            with contextlib.suppress(Exception):
                store.activate_for_paths(paths)

    async def _maybe_extract_session_memory(self, messages: list[Message]) -> None:
        """Opportunistically update durable session memory after a clean turn.

        Extraction must never block the user-visible loop from finishing, so the
        work is launched as a best-effort background task.  The manager itself
        serializes concurrent extractions for the same session.
        """
        if not self._session_memory:
            return
        try:
            should_extract = await self._session_memory.should_extract(messages)
        except Exception:
            return
        if not should_extract:
            return

        snapshot = list(messages)

        async def _run() -> None:
            with contextlib.suppress(Exception):
                await self._session_memory.extract(model=self._model, messages=snapshot)

        asyncio.create_task(_run(), name=f"session-memory-{self._config.session_id}")

    async def _stream_with_idle_timeout(
        self,
        stream: AsyncIterator[StreamDelta],
        *,
        turn: int,
    ) -> AsyncIterator[StreamDelta]:
        """逐个读取模型 delta，并对无响应的 provider 做 idle timeout。

        没有这个保护时，工具完成后下一轮模型请求如果卡在网络/网关层，
        前端只会停在最后一个 ✓ tool_completed，永远等不到 cursor。
        """
        iterator = stream.__aiter__()
        timeout = self._config.model_stream_idle_timeout_seconds
        if timeout <= 0:
            async for delta in iterator:
                yield delta
            return
        while True:
            try:
                delta = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                with contextlib.suppress(Exception):
                    await iterator.aclose()  # type: ignore[attr-defined]
                raise TimeoutError(
                    f"model stream idle timeout after {timeout:g}s on turn {turn}; "
                    "the model/provider did not return any delta after the previous event"
                ) from exc
            yield delta

    async def _fire_post_tool_use(
        self, tu: ToolUseBlock, result_content: str,
    ) -> AsyncIterator[AgentEvent]:
        """触发 PostToolUse hook 并产出事件。"""
        post_hook = await self._hook_engine.fire_post_tool_use(
            tu.name, tu.input,
            session_id=self._config.session_id,
            cwd=self._config.cwd,
            tool_use_id=tu.id,
        )
        for msg in post_hook.system_messages:
            yield AgentEvent("hook_message", {"message": msg})
        for ctx in post_hook.additional_contexts:
            yield AgentEvent("hook_message", {"message": ctx})

    async def _process_tool_failure(self, tu: ToolUseBlock, result: ToolResult) -> AsyncIterator[AgentEvent]:
        """分类工具失败、触发恢复 hook，并产出 Terminal/UI 可见恢复事件。"""
        failure = recovery_hint_for_tool_result(tu.name, result.content, result.metadata)
        result.metadata.update(failure.to_metadata())
        key = json.dumps({"tool": tu.name, "input": tu.input, "category": failure.category.value, "content": result.content[:240]}, sort_keys=True, default=str)
        count = self._tool_failure_counts.get(key, 0) + 1
        self._tool_failure_counts[key] = count
        payload = {
            "tool_use_id": tu.id,
            "tool": tu.name,
            "message": failure.message,
            "repeat_count": count,
            **failure.to_metadata(),
        }
        yield AgentEvent("tool_failure", payload)
        yield AgentEvent("recovery_suggested", payload)
        if count >= 2:
            hint = repeated_failure_hint(tu.name, result.content, count=count)
            result.metadata["repeatedFailureHint"] = hint
            yield AgentEvent("recovery_suggested", {**payload, "repeated": True, "recoveryHint": hint})

        post_failure = await self._hook_engine.fire_post_tool_use_failure(
            tu.name,
            tu.input,
            session_id=self._config.session_id,
            cwd=self._config.cwd,
            tool_use_id=tu.id,
            error=result.content,
            category=failure.category.value,
            recovery_hint=failure.hint,
        )
        for msg in post_failure.system_messages:
            yield AgentEvent("hook_message", {"message": msg})
        if post_failure.should_block:
            yield AgentEvent("hook_message", {"message": f"PostToolUseFailure blocked continuation: {post_failure.stop_reason}"})
        recovery_hook = await self._hook_engine.fire_recovery_suggested(
            tu.name,
            tu.input,
            session_id=self._config.session_id,
            cwd=self._config.cwd,
            tool_use_id=tu.id,
            category=failure.category.value,
            recovery_hint=failure.hint,
        )
        for msg in recovery_hook.system_messages:
            yield AgentEvent("hook_message", {"message": msg})

    async def _fire_permission_denied_and_recovery(
        self,
        tu: ToolUseBlock,
        reason: str,
        failure: Any,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent("recovery_suggested", {"tool_use_id": tu.id, "tool": tu.name, "message": reason, **failure.to_metadata()})
        denied = await self._hook_engine.fire_permission_denied(
            tu.name,
            tu.input,
            session_id=self._config.session_id,
            cwd=self._config.cwd,
            tool_use_id=tu.id,
            reason=reason,
        )
        for msg in denied.system_messages:
            yield AgentEvent("hook_message", {"message": msg})
        if denied.system_messages:
            yield AgentEvent("permission_retry_available", {"tool_use_id": tu.id, "tool": tu.name, "reason": reason})
