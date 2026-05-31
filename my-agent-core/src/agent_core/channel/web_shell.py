"""WebShell 渠道 — WebSocket + xterm.js 实现。

权限交互流程：
1. 后端发送 JSON: {"type": "permission_request", "tool": "read_text_file", "input": {...}, "reason": "..."}
2. 前端渲染浮动卡片（工具名 + 输入详情 + Allow/Deny 按钮）
3. 用户点击按钮，前端发送 JSON: {"type": "permission_response", "decision": "allow"|"deny"}
4. 后端 resolve future，继续 agent loop

也兼容旧式终端输入（Y/n），方便纯终端调试。
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from agent_core.adapters.web import event_to_terminal_text
from agent_core.channel.base import (
    Channel,
    make_permission_request_message,
    make_permission_result_message,
    parse_permission_response,
)
from agent_core.core.events import AgentEvent


class WebShellChannel:
    """WebSocket + xterm.js 渠道。

    参数:
        send_func: async callable，向 WebSocket 发送文本
        on_permission_response: async callable，等待用户权限决策
            返回 "allow" 或 "deny"
    """

    def __init__(
        self,
        *,
        send_func: Callable[[str], Awaitable[None]],
        wait_permission_func: Callable[[], Awaitable[str]],
    ) -> None:
        self._send_func = send_func
        self._wait_permission = wait_permission_func

    async def send(self, text: str) -> None:
        """发送纯文本到 WebSocket。"""
        await self._send_func(text)

    async def send_event(self, event: AgentEvent) -> None:
        """将 AgentEvent 渲染为终端文本后发送。"""
        text = event_to_terminal_text(event)
        if text:
            await self._send_func(text)

    async def ask_permission(self, tool_name: str, tool_input: dict, reason: str) -> str:
        """向用户请求权限 — 发送 JSON 卡片，等待前端按钮响应。"""
        # 1. 发送 JSON 结构化消息（前端渲染弹窗）
        msg = make_permission_request_message(tool_name, tool_input, reason)
        await self._send_func(msg)

        # 2. 等待前端响应（由 app.py 的 WebSocket handler resolve future）
        decision = await self._wait_permission()

        # 3. 发送结果确认
        result_msg = make_permission_result_message(tool_name, decision)
        await self._send_func(result_msg)

        return decision
