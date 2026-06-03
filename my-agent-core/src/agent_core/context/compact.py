"""上下文压缩 — 借鉴 Claude Code 的 compact 系统。

三级压缩策略：
1. MicroCompact: 清除旧 tool_result 内容（最快，零 API 调用）
2. FullCompact: LLM 生成摘要（最慢但最完整）
3. Auto-compact: 在 agent loop 中自动触发

设计原则：
- 压缩是破坏性操作，替换 messages 内容后不可逆
- 摘要消息注入后，agent 应能无缝继续工作
- 熔断机制：连续失败后停止尝试
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.context.tokens import estimate_messages_tokens, estimate_block_tokens, token_count_with_estimation
from agent_core.model.base import ModelClient, StreamDelta
from agent_core.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    ThinkingBlock,
    system_prompt_to_str,
    SystemPrompt,
)


# ── 常量 ──────────────────────────────────────────────────────

MICRO_COMPACT_CLEARED = "[Old tool result content cleared]"
"""MicroCompact 替换旧 tool_result 的占位文本。"""

AUTOCOMPACT_BUFFER_TOKENS = 13_000
"""自动压缩预留 buffer（借鉴 Claude Code 的 AUTOCOMPACT_BUFFER_TOKENS）。"""

MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
"""连续自动压缩失败熔断阈值。"""


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


def _default_compact_stream_idle_timeout_seconds() -> float:
    return _env_float(
        (
            "AGENT_COMPACT_STREAM_IDLE_TIMEOUT_SECONDS",
            "AGENT_MODEL_STREAM_IDLE_TIMEOUT_SECONDS",
            "COMPACT_STREAM_IDLE_TIMEOUT_SECONDS",
            "MODEL_STREAM_IDLE_TIMEOUT_SECONDS",
        ),
        120.0,
    )

COMPACT_PROMPT = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
- Do NOT use any tool. You already have all the context you need.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

Analyze the conversation above and produce a structured summary with these sections:

1. **Primary Request and Intent**: What the user asked for and their goals
2. **Key Technical Concepts**: Important concepts, patterns, and architecture decisions
3. **Files and Code Sections**: Key files and code that was read or modified (include code snippets)
4. **Errors and Fixes**: Any errors encountered and how they were fixed
5. **Problem Solving**: How problems were approached and resolved
6. **All User Messages**: List all non-tool-result user messages
7. **Pending Tasks**: Any tasks that are still in progress or not yet started
8. **Current Work**: What was being worked on most recently (be detailed)
9. **Optional Next Step**: What should be done next to continue the work

Format:
<analysis>
[Brief analysis of the conversation flow and key decisions]
</analysis>

<summary>
1. **Primary Request and Intent**: ...
2. **Key Technical Concepts**: ...
3. **Files and Code Sections**: ...
4. **Errors and Fixes**: ...
5. **Problem Solving**: ...
6. **All User Messages**: ...
7. **Pending Tasks**: ...
8. **Current Work**: ...
9. **Optional Next Step**: ...
</summary>
"""

COMPACT_RESUME_PREFIX = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
)
COMPACT_RESUME_SUFFIX = (
    "\n\nContinue the conversation from where it left off without asking the user any further questions. "
    "Resume directly — do not acknowledge the summary, do not recap what was happening, "
    "do not preface with \"I'll continue\" or similar. Pick up the last task as if the break never happened."
)


# ── MicroCompact ──────────────────────────────────────────────


@dataclass(slots=True)
class MicroCompactConfig:
    """MicroCompact 配置。"""
    keep_recent: int = 3
    """保留最近 N 个 tool_result 不清除。"""

    compactable_tools: set[str] = field(default_factory=lambda: {
        "echo", "read_text_file", "bash", "grep", "glob",
        "web_search", "web_fetch", "file_edit", "file_write",
    })
    """可清除内容的工具集合。"""


def micro_compact(
    messages: list[Message],
    config: MicroCompactConfig | None = None,
) -> list[Message]:
    """MicroCompact: 清除旧 tool_result 内容，保留最近 N 个。

    借鉴 Claude Code 的 time-based microCompact：
    - 找到所有 compactable 工具的 tool_result
    - 保留最近 keep_recent 个
    - 其余替换为 MICRO_COMPACT_CLEARED

    注意：此函数修改消息内容（因为缓存已冷，无需保留原文）。
    返回新的消息列表（不修改原列表）。
    """
    cfg = config or MicroCompactConfig()

    # 1. 收集所有 compactable tool_result 的 (msg_idx, block_idx)
    compactable_ids: list[tuple[int, int]] = []
    for mi, msg in enumerate(messages):
        for bi, block in enumerate(msg.content):
            if isinstance(block, ToolResultBlock):
                # 检查对应的 tool_use 是否在 compactable_tools 中
                if _is_compactable_tool_result(block, messages, mi, cfg.compactable_tools):
                    compactable_ids.append((mi, bi))

    if not compactable_ids:
        return messages

    # 2. 保留最近 keep_recent 个
    keep_set = set(compactable_ids[-cfg.keep_recent:])
    clear_set = set(compactable_ids) - keep_set

    if not clear_set:
        return messages

    # 3. 替换旧 tool_result 内容
    new_messages = []
    for mi, msg in enumerate(messages):
        new_blocks = list(msg.content)
        changed = False
        for bi, block in enumerate(new_blocks):
            if (mi, bi) in clear_set and isinstance(block, ToolResultBlock):
                new_blocks[bi] = ToolResultBlock(
                    tool_use_id=block.tool_use_id,
                    content=MICRO_COMPACT_CLEARED,
                    is_error=block.is_error,
                    metadata=block.metadata,
                )
                changed = True
        if changed:
            new_messages.append(Message(role=msg.role, content=new_blocks, metadata=msg.metadata))
        else:
            new_messages.append(msg)

    return new_messages


def _is_compactable_tool_result(
    block: ToolResultBlock,
    messages: list[Message],
    msg_idx: int,
    compactable_tools: set[str],
) -> bool:
    """判断 tool_result 是否来自 compactable 工具。

    通过 tool_use_id 回溯找到对应的 ToolUseBlock，检查 tool name。
    """
    tool_use_id = block.tool_use_id
    # 在 msg_idx 之前的消息中查找对应的 tool_use
    for mi in range(msg_idx, -1, -1):
        for b in messages[mi].content:
            if isinstance(b, ToolUseBlock) and b.id == tool_use_id:
                return b.name in compactable_tools
    # 找不到对应 tool_use，默认可 compact
    return True


# ── FullCompact ───────────────────────────────────────────────


async def full_compact(
    messages: list[Message],
    model: ModelClient,
    *,
    system: str | SystemPrompt | None = None,
    custom_instructions: str | None = None,
    max_output_tokens: int = 4096,
    stream_idle_timeout_seconds: float = 120.0,
) -> list[Message]:
    """FullCompact: 用 LLM 生成对话摘要，替换历史消息。

    借鉴 Claude Code 的 compactConversation：
    1. 构建摘要请求（COMPACT_PROMPT + 对话历史）
    2. 流式生成摘要
    3. 格式化摘要（剥离 <analysis> 块，保留 <summary>）
    4. 构建摘要消息替换历史

    返回只包含摘要消息的新消息列表。
    """
    system_str = system if isinstance(system, str) else (
        system_prompt_to_str(system) if system else "You are a helpful assistant."
    )

    # 1. 构建摘要请求
    compact_messages = _build_compact_messages(messages, custom_instructions=custom_instructions)

    # 2. 流式生成摘要
    summary_text = ""
    stream = model.stream(
        system=system_str,
        messages=compact_messages,
        tools=[],  # 摘要请求不使用工具
        metadata={"max_tokens": max_output_tokens},
    )
    async for delta in _stream_with_idle_timeout(stream, timeout=stream_idle_timeout_seconds):
        if delta.type == "text_delta":
            summary_text += delta.text

    # 3. 格式化摘要
    formatted = format_compact_summary(summary_text)
    if not formatted:
        raise ValueError("Failed to generate conversation summary.")

    # 4. 构建摘要消息
    resume_text = COMPACT_RESUME_PREFIX + formatted + COMPACT_RESUME_SUFFIX
    summary_message = Message.user(resume_text)
    summary_message.metadata.update({
        "compact_summary": True,
        "pre_compact_message_count": len(messages),
        "summary": formatted,
    })

    return [summary_message]


async def _stream_with_idle_timeout(
    stream: AsyncIterator[StreamDelta],
    *,
    timeout: float,
) -> AsyncIterator[StreamDelta]:
    """为 full compact 摘要请求增加 idle timeout，避免压缩阶段无限等待。"""
    iterator = stream.__aiter__()
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
            raise TimeoutError(f"compact model stream idle timeout after {timeout:g}s") from exc
        yield delta


def _build_compact_messages(messages: list[Message], *, custom_instructions: str | None = None) -> list[Message]:
    """构建用于摘要请求的消息列表。

    在对话历史前插入摘要指令消息。
    剥离图片等大体积内容（如果有的话）。
    """
    # 插入摘要指令
    prompt = COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += "\n\nAdditional summarization instructions from the user:\n" + custom_instructions.strip()
    instruction = Message.user(prompt)

    # 简化消息：剥离 thinking blocks（减少 token 消耗）
    simplified = []
    for msg in messages:
        new_blocks = []
        for block in msg.content:
            if isinstance(block, ThinkingBlock):
                continue  # 跳过 thinking blocks
            if isinstance(block, ToolResultBlock) and block.content == MICRO_COMPACT_CLEARED:
                continue  # 跳过已清除的 tool_result
            new_blocks.append(block)
        if new_blocks:
            simplified.append(Message(role=msg.role, content=new_blocks, metadata=msg.metadata))

    return simplified + [instruction]


def format_compact_summary(raw_summary: str) -> str:
    """格式化 LLM 生成的摘要。

    借鉴 Claude Code 的 formatCompactSummary：
    - 剥离 <analysis> 块（仅作为起草草稿）
    - 将 <summary> 标签替换为 Summary: 标题
    """
    import re

    # 剥离 <analysis>...</analysis>
    result = re.sub(r"<analysis>.*?</analysis>", "", raw_summary, flags=re.DOTALL).strip()

    # 替换 <summary>...</summary> 标签
    result = re.sub(r"</?summary>", "", result).strip()

    # 如果结果为空，返回原始摘要
    if not result:
        return raw_summary.strip()

    return result


# ── AutoCompact ───────────────────────────────────────────────


@dataclass(slots=True)
class ManualCompactResult:
    """手动 compact 的结果元信息。"""
    messages: list[Message]
    summary: str
    pre_message_count: int
    post_message_count: int
    pre_token_count: int
    post_token_count: int


@dataclass(slots=True)
class AutoCompactConfig:
    """自动压缩配置。"""
    context_window: int = 200_000
    """模型上下文窗口大小（tokens）。"""

    reserved_for_output: int = 20_000
    """为模型输出预留的 token 数。"""

    buffer_tokens: int = AUTOCOMPACT_BUFFER_TOKENS
    """触发自动压缩的 buffer。"""

    enable_micro: bool = True
    """是否启用 micro compact。"""

    enable_full: bool = True
    """是否启用 full compact（LLM 摘要）。"""

    micro_config: MicroCompactConfig = field(default_factory=MicroCompactConfig)

    max_consecutive_failures: int = MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES

    compact_stream_idle_timeout_seconds: float = field(default_factory=_default_compact_stream_idle_timeout_seconds)
    """Full compact 摘要请求两次 delta 之间的最大空闲时间；<=0 表示禁用 idle timeout。"""

    @property
    def effective_context_window(self) -> int:
        """有效上下文窗口 = 总窗口 - 输出预留。"""
        return self.context_window - self.reserved_for_output

    @property
    def auto_compact_threshold(self) -> int:
        """自动压缩触发阈值 = 有效窗口 - buffer。"""
        return self.effective_context_window - self.buffer_tokens


class AutoCompactor:
    """自动压缩管理器。

    借鉴 Claude Code 的 autoCompactIfNeeded：
    - 每轮结束后检查 token 数是否超过阈值
    - 先尝试 micro_compact（轻量）
    - 如果仍然超限，走 full_compact（LLM 摘要）
    - 熔断：连续失败超过阈值后停止尝试
    """

    def __init__(self, config: AutoCompactConfig | None = None) -> None:
        self._config = config or AutoCompactConfig()
        self._consecutive_failures = 0
        self._last_exact_count: int | None = None
        self._last_exact_index: int = -1

    def update_exact_count(self, prompt_tokens: int, message_index: int) -> None:
        """更新精确的 API token 计数。"""
        self._last_exact_count = prompt_tokens
        self._last_exact_index = message_index

    def should_compact(self, messages: list[Message]) -> bool:
        """判断是否需要自动压缩。"""
        if self._consecutive_failures >= self._config.max_consecutive_failures:
            return False

        token_count = token_count_with_estimation(
            messages,
            last_exact_count=self._last_exact_count,
            last_exact_index=self._last_exact_index,
        )
        return token_count >= self._config.auto_compact_threshold

    async def compact_if_needed(
        self,
        messages: list[Message],
        model: ModelClient,
        *,
        system: str | SystemPrompt | None = None,
        force: bool = False,
    ) -> tuple[list[Message], bool]:
        """如果需要，执行自动压缩。

        返回 (新消息列表, 是否执行了压缩)。
        """
        if not force and not self.should_compact(messages):
            return messages, False

        # 1. 先尝试 micro_compact
        if self._config.enable_micro:
            new_messages = micro_compact(messages, self._config.micro_config)
            if not self.should_compact(new_messages):
                self._consecutive_failures = 0
                return new_messages, True

        # 2. micro 不够，走 full_compact
        if self._config.enable_full:
            try:
                new_messages = await full_compact(
                    messages if not self._config.enable_micro else micro_compact(messages, self._config.micro_config),
                    model,
                    system=system,
                    stream_idle_timeout_seconds=self._config.compact_stream_idle_timeout_seconds,
                )
                self._consecutive_failures = 0
                self._last_exact_count = None  # 重置精确计数
                return new_messages, True
            except Exception:
                self._consecutive_failures += 1
                return messages, False

        return messages, False

    def token_count_with_estimation(self, messages: list[Message]) -> int:
        """使用混合精确+估算计算当前 token 数。"""
        from agent_core.context.tokens import token_count_with_estimation as _tce
        return _tce(messages, self._last_exact_count, self._last_exact_index)

    def reset_failures(self) -> None:
        """重置连续失败计数。"""
        self._consecutive_failures = 0


async def manual_compact(
    messages: list[Message],
    model: ModelClient,
    *,
    system: str | SystemPrompt | None = None,
    custom_instructions: str | None = None,
    max_output_tokens: int = 4096,
    stream_idle_timeout_seconds: float = 120.0,
) -> ManualCompactResult:
    """手动 /compact：生成摘要并替换当前会话历史。

    对齐 Claude Code 的核心语义：清掉旧 conversation history，但把一条
    compact summary 留在 context 中，让下一轮可以无缝继续。
    """
    if not messages:
        raise ValueError("Not enough messages to compact.")

    pre_message_count = len(messages)
    pre_token_count = token_count_with_estimation(messages)
    compacted = await full_compact(
        messages,
        model,
        system=system,
        custom_instructions=custom_instructions,
        max_output_tokens=max_output_tokens,
        stream_idle_timeout_seconds=stream_idle_timeout_seconds,
    )
    summary = _extract_summary_text(compacted)
    post_token_count = token_count_with_estimation(compacted)
    return ManualCompactResult(
        messages=compacted,
        summary=summary,
        pre_message_count=pre_message_count,
        post_message_count=len(compacted),
        pre_token_count=pre_token_count,
        post_token_count=post_token_count,
    )


def _extract_summary_text(messages: list[Message]) -> str:
    for msg in messages:
        if msg.metadata.get("compact_summary"):
            texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            return "\n".join(texts)
    return ""
