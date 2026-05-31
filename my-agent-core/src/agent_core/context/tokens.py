"""Token 估算 — 借鉴 Claude Code 的 tokenEstimation.ts。

核心策略：
- 精确计数：优先使用 API 返回的 usage.prompt_tokens
- 粗略估算：length / bytes_per_token（默认 4，JSON 2）
- 混合模式：从最近精确计数 + 后续消息粗略估算
"""
from __future__ import annotations

import json
from typing import Any

from agent_core.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock


# ── 粗略估算 ──────────────────────────────────────────────────


def rough_token_count(text: str, bytes_per_token: int = 4) -> int:
    """粗略 token 估算：字符数 / bytes_per_token。"""
    return max(1, len(text) // bytes_per_token)


def bytes_per_token_for_content(content: str, is_json: bool = False) -> int:
    """根据内容类型返回 chars/token 比率。JSON 密集编码比率约 2。"""
    return 2 if is_json else 4


# ── 单 block 估算 ─────────────────────────────────────────────


def estimate_block_tokens(block: Any) -> int:
    """估算单个 ContentBlock 的 token 数。"""
    if isinstance(block, TextBlock):
        return rough_token_count(block.text)
    elif isinstance(block, ThinkingBlock):
        return rough_token_count(block.thinking)
    elif isinstance(block, ToolUseBlock):
        # name + JSON(input)
        payload = block.name + json.dumps(block.input, ensure_ascii=False)
        return rough_token_count(payload, bytes_per_token=2)
    elif isinstance(block, ToolResultBlock):
        return rough_token_count(block.content)
    else:
        # fallback: 整体序列化
        return rough_token_count(json.dumps(str(block), ensure_ascii=False), bytes_per_token=2)


# ── 消息级别估算 ──────────────────────────────────────────────


def estimate_message_tokens(message: Message) -> int:
    """估算单条 Message 的 token 数（含 role 开销约 4 token）。"""
    total = 4  # role + formatting overhead
    for block in message.content:
        total += estimate_block_tokens(block)
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    """估算消息列表的总 token 数。"""
    return sum(estimate_message_tokens(m) for m in messages)


# ── 混合精确 + 估算 ──────────────────────────────────────────


def token_count_with_estimation(
    messages: list[Message],
    last_exact_count: int | None = None,
    last_exact_index: int = -1,
) -> int:
    """混合精确计数 + 粗略估算。

    借鉴 Claude Code 的 tokenCountWithEstimation：
    - 如果有精确的 API usage 数据，从该点开始只估算后续消息
    - 否则全量粗略估算

    参数:
        messages: 消息列表
        last_exact_count: 最近一次 API 返回的 prompt_tokens
        last_exact_index: 对应精确计数的消息列表索引（-1 表示最后一条）
    """
    if last_exact_count is not None and messages:
        idx = last_exact_index if last_exact_index >= 0 else len(messages) - 1
        # 精确计数覆盖 [0..idx]，估算 [idx+1..]
        subsequent = messages[idx + 1:]
        return last_exact_count + estimate_messages_tokens(subsequent)
    return estimate_messages_tokens(messages)
