from __future__ import annotations

import hashlib

from agent_core.buddy.types import Companion
from agent_core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def observe_companion_reaction(messages: list[Message], companion: Companion | None) -> str | None:
    if companion is None or companion.muted or not messages:
        return None
    last_user = _last_user_text(messages)
    last_tool_error = _last_tool_error(messages)
    last_tool = _last_tool_name(messages)
    lower = last_user.lower()
    name_hit = companion.name.lower() in lower if companion.name else False
    if name_hit:
        return _pick(companion, [
            f"{companion.name}听见了：我在旁边盯着边界条件。",
            "我来负责眨眼和怀疑，你继续负责主线。",
            "收到，我会小声提醒，不抢主模型的戏。",
        ], salt=last_user)
    if last_tool_error:
        return _pick(companion, ["这个工具摔了一跤。", "先别急，错误信息通常比它看起来更诚实。", "我闻到了一个可复现的 bug。"], salt=last_tool_error)
    if last_tool in {"write_text_file", "edit_file", "bash", "TodoWrite"}:
        return _pick(companion, ["动过刀了，记得跑验证。", "我看到改动了。小心、漂亮、可回滚。", "这一步像是真的在推进。"], salt=last_tool + last_user)
    if _stable_chance(last_user + companion.name, 0.28):
        return _pick(companion, ["我在。", "这个方向闻起来还不错。", "边界条件没有睡着，我也没有。", "继续，我负责在角落闪光。"], salt=last_user)
    return None


def _last_user_text(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        parts = [block.text for block in msg.content if isinstance(block, TextBlock)]
        if parts:
            return "\n".join(parts)
    return ""


def _last_tool_error(messages: list[Message]) -> str:
    for msg in reversed(messages):
        for block in msg.content:
            if isinstance(block, ToolResultBlock) and block.is_error:
                return block.content
    return ""


def _last_tool_name(messages: list[Message]) -> str:
    for msg in reversed(messages):
        for block in reversed(msg.content):
            if isinstance(block, ToolUseBlock):
                return block.name
    return ""


def _stable_chance(seed: str, threshold: float) -> bool:
    value = int.from_bytes(hashlib.blake2s(seed.encode("utf-8"), digest_size=4).digest(), "big") / 2**32
    return value < threshold


def _pick(companion: Companion, values: list[str], *, salt: str) -> str:
    idx = int.from_bytes(hashlib.blake2s(f"{companion.name}:{salt}".encode("utf-8"), digest_size=2).digest(), "big") % len(values)
    return values[idx]
