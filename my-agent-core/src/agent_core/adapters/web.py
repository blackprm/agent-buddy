from __future__ import annotations

import json
from collections.abc import AsyncIterator

from agent_core.core.agent import AgentRuntime
from agent_core.core.events import AgentEvent
from agent_core.types import Message


def event_to_sse(event: AgentEvent) -> str:
    """把 AgentEvent 转成 Server-Sent Events 字符串。"""

    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.type}\ndata: {payload}\n\n"


async def agent_sse(runtime: AgentRuntime, user_input: str, history: list[Message] | None = None) -> AsyncIterator[str]:
    async for event in runtime.run(user_input, history=history):
        yield event_to_sse(event)


def event_to_terminal_text(event: AgentEvent) -> str:
    """把 AgentEvent 转成终端可显示的 ANSI 文本。用于 PTY 桥接。"""

    if event.type == "text_delta":
        return event.data.get("text", "")
    elif event.type == "thinking_delta":
        text = event.data.get("text", "")
        return f"\x1b[90m{text}\x1b[0m" if text else ""
    elif event.type == "assistant_text":
        return ""
    elif event.type == "tool_started":
        tool = event.data.get("tool", "?")
        concurrent = " (concurrent)" if event.data.get("concurrent") else ""
        return f"\r\n\x1b[33m▸ {tool}{concurrent}\x1b[0m"
    elif event.type == "tool_progress":
        if event.data.get("tool") == "bash":
            elapsed = event.data.get("elapsedTimeSeconds")
            output = (event.data.get("output") or "").strip()
            suffix = f" — {elapsed}s" if elapsed is not None else ""
            preview = f"\r\n\x1b[90m{output[-500:]}\x1b[0m" if output else ""
            return f"\r\n\x1b[36m[bash running{suffix}]\x1b[0m{preview}"
        return ""
    elif event.type == "tool_completed":
        tool = event.data.get("tool", "?")
        is_error = event.data.get("is_error", False)
        color = "\x1b[31m" if is_error else "\x1b[32m"
        result_content = event.data.get("result", "")
        if result_content:
            preview = result_content if len(result_content) <= 500 else result_content[:500] + "..."
            return f"{color}✓ {tool}\x1b[0m\r\n\x1b[90m{preview}\x1b[0m\r\n"
        return f"{color}✓ {tool}\x1b[0m\r\n"
    elif event.type == "tool_denied":
        tool = event.data.get("tool", "?")
        reason = event.data.get("reason", "")
        return f"\r\n\x1b[31m✗ {tool}: {reason}\x1b[0m\r\n"
    elif event.type == "tool_failure":
        tool = event.data.get("tool", "?")
        category = event.data.get("failureCategory", "tool_failure")
        return f"\r\n\x1b[31m[tool failure] {tool}: {category}\x1b[0m\r\n"
    elif event.type == "recovery_suggested":
        hint = event.data.get("recoveryHint") or event.data.get("message") or ""
        return f"\x1b[33m[recovery] {hint}\x1b[0m\r\n" if hint else ""
    elif event.type == "api_retry":
        attempt = event.data.get("attempt", "?")
        max_retries = event.data.get("max_retries", "?")
        delay = event.data.get("retry_delay_ms", 0)
        category = event.data.get("category", "api_error")
        return f"\r\n\x1b[33m[api retry] {category} — retrying in {delay}ms ({attempt}/{max_retries})\x1b[0m\r\n"
    elif event.type == "model_error":
        return f"\r\n\x1b[31m[model error] {event.data.get('category', 'unknown')}: {event.data.get('error', '')}\x1b[0m\r\n"
    elif event.type == "conversation_repaired":
        return "\r\n\x1b[33m[conversation repaired: fixed tool_use/tool_result pairing]\x1b[0m\r\n"
    elif event.type == "permission_retry_available":
        tool = event.data.get("tool", "?")
        return f"\x1b[33m[permission retry available] {tool}\x1b[0m\r\n"
    elif event.type == "workspace_state":
        status = event.data.get("status", "unknown")
        return f"\r\n\x1b[36m[isolated workspace: {status}]\x1b[0m\r\n"
    elif event.type == "permission_request":
        return ""
    elif event.type == "context_compacting":
        msg_count = event.data.get("message_count", "?")
        return f"\r\n\x1b[36m[compacting context — {msg_count} messages]\x1b[0m\r\n"
    elif event.type == "context_compacted":
        msg_count = event.data.get("message_count", "?")
        return f"\r\n\x1b[36m[context compacted — {msg_count} messages remaining]\x1b[0m\r\n"
    elif event.type == "context_compact_failed":
        return "\r\n\x1b[33m[context compact failed — continuing with existing context]\x1b[0m\r\n"
    elif event.type == "hook_message":
        message = event.data.get("message", "")
        return f"\x1b[35m[hook] {message}\x1b[0m\r\n"
    elif event.type == "loop_completed":
        return "\r\n"
    elif event.type == "loop_failed":
        error = event.data.get("error", "unknown")
        return f"\r\n\x1b[31merror: {error}\x1b[0m\r\n"
    elif event.type == "loop_aborted":
        return "\r\n\x1b[33m[interrupted]\x1b[0m\r\n"
    elif event.type in ("model_started", "model_completed", "turn_started", "loop_started"):
        return ""
    return ""


# ── 聊天 UI 事件 ──────────────────────────────────────────────

# 需要发送给聊天 UI 的事件（JSON 格式）
_CHAT_EVENTS = {
    "model_started", "model_completed",
    "text_delta", "thinking_delta", "assistant_text",
    "tool_started", "tool_progress", "tool_completed", "tool_denied",
    "permission_request", "permission_result", "permission_retry_available", "agent_status",
    "context_compacting", "context_compacted", "context_compact_failed", "hook_message", "companion_reaction", "task_state",
    "workspace_state", "plan_implementation_started",
    "tool_failure", "recovery_suggested", "api_retry", "model_error", "conversation_repaired",
    "loop_completed", "loop_failed", "loop_aborted",
}


def event_to_chat_json(event: AgentEvent) -> str | None:
    """把 AgentEvent 转成聊天 UI 的 JSON 消息。返回 None 表示不需要发送。"""

    if event.type not in _CHAT_EVENTS:
        return None

    payload = {
        "type": event.type,
        **event.data,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)
