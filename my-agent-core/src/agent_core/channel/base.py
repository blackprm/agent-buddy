"""渠道抽象 — Channel Protocol。

开闭原则：
- Channel Protocol：稳定的抽象接口，不随渠道变化
- 新增渠道只需实现 Protocol，注册到 ChannelRegistry
- AgentRuntime 通过 Channel 与用户交互，不关心底层是 WebSocket 还是飞书

WebSocket 消息协议：
- 普通输出：纯文本（ANSI 转义序列）
- 结构化消息：JSON，以 {"type": "xxx", ...} 格式
  - permission_request: 权限询问卡片
  - permission_response: 用户权限决策
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from agent_core.core.events import AgentEvent


# ── Channel Protocol ─────────────────────────────────────────


@runtime_checkable
class Channel(Protocol):
    """渠道接口 — 对扩展开放，对修改关闭。

    每个渠道负责：
    1. 将 AgentEvent 渲染并发送给用户
    2. 处理权限询问交互（ask_permission）
    3. 接收用户输入（由具体渠道自行管理）

    新增渠道（飞书、Slack、Discord...）只需实现此 Protocol。
    """

    async def send(self, text: str) -> None:
        """发送文本内容到渠道。"""
        ...

    async def send_event(self, event: AgentEvent) -> None:
        """将 AgentEvent 渲染后发送到渠道。"""
        ...

    async def ask_permission(self, tool_name: str, tool_input: dict, reason: str) -> str:
        """向用户请求权限，返回 'allow' 或 'deny'。

        渠道决定如何展示询问 UI：
        - WebShell: JSON 消息 → 前端弹窗 → 按钮点击
        - 飞书: 交互卡片 → 按钮回调
        """
        ...


# ── 渠道注册表 ────────────────────────────────────────────────

_CHANNEL_REGISTRY: dict[str, type] = {}


def register_channel(name: str, cls: type) -> None:
    """注册渠道实现。"""
    _CHANNEL_REGISTRY[name.lower()] = cls


def create_channel(channel_type: str, **kwargs: Any) -> Channel:
    """工厂方法：根据类型名创建 Channel 实例。"""
    _ensure_builtin_registered()
    cls = _CHANNEL_REGISTRY.get(channel_type.lower())
    if cls is None:
        available = ", ".join(sorted(_CHANNEL_REGISTRY.keys())) or "none"
        raise ValueError(f"Unknown channel type '{channel_type}'. Available: {available}")
    return cls(**kwargs)


_builtin_registered = False


def _ensure_builtin_registered() -> None:
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True
    from agent_core.channel.web_shell import WebShellChannel
    register_channel("web_shell", WebShellChannel)


# ── WebSocket JSON 消息协议 ──────────────────────────────────


def make_permission_request_message(
    tool_name: str,
    tool_input: dict,
    reason: str,
    options: list[dict] | None = None,
) -> str:
    """构造权限询问的 JSON 消息。"""
    return json.dumps({
        "type": "permission_request",
        "tool": tool_name,
        "input": tool_input,
        "reason": reason,
        "options": options or [],
    }, ensure_ascii=False)


def make_permission_result_message(
    tool_name: str,
    decision: str,
) -> str:
    """构造权限结果的 JSON 消息（发给前端确认用）。"""
    return json.dumps({
        "type": "permission_result",
        "tool": tool_name,
        "decision": decision,
    }, ensure_ascii=False)


def parse_permission_response(data: str) -> dict | None:
    """解析前端发回的权限决策。返回 decision + option 或 None。"""
    try:
        msg = json.loads(data)
        if isinstance(msg, dict) and msg.get("type") == "permission_response":
            decision = msg.get("decision", "deny")
            option = msg.get("option") if isinstance(msg.get("option"), dict) else None
            return {"decision": "allow" if decision == "allow" else "deny", "option": option}
    except (json.JSONDecodeError, TypeError):
        pass
    return None
