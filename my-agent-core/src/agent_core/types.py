from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NewType


Role = Literal["user", "assistant", "system"]

# ── System Prompt 类型（借鉴 Claude Code 的 branded type + section 架构）──

CacheScope = Literal["global", "org"]
"""Prompt cache 作用域：
- global: 跨用户/跨组织缓存（仅静态内容）
- org: 组织级缓存
未指定时为 None，表示不缓存。
"""

DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
"""静态/动态内容边界标记。边界之前的 section 可用 global scope 缓存，
边界之后的内容是 per-session 的，不应跨用户缓存。"""


@dataclass(slots=True)
class SystemPromptBlock:
    """带缓存作用域的 system prompt 文本块。用于 API 调用时分配 cache_control。"""
    text: str
    cache_scope: CacheScope | None = None


SystemPrompt = NewType("SystemPrompt", list[str])
"""品牌类型：system prompt 由有序 section 列表组成，不是单个字符串。

通过 NewType 确保不会与普通 list[str] 混淆，必须显式构造。
工厂函数 as_system_prompt() 是唯一的构造入口。"""


def as_system_prompt(sections: list[str]) -> SystemPrompt:
    """SystemPrompt 的唯一构造工厂。"""
    return SystemPrompt(sections)


def system_prompt_to_str(prompt: SystemPrompt) -> str:
    """将 SystemPrompt 展平为单个字符串（用于不支持多 system block 的 API）。"""
    return "\n\n".join(s for s in prompt if s)


def system_prompt_to_blocks(
    prompt: SystemPrompt,
    *,
    enable_caching: bool = False,
    skip_global_cache: bool = False,
) -> list[SystemPromptBlock]:
    """将 SystemPrompt 拆分为带缓存作用域的 SystemPromptBlock 列表。

    拆分逻辑（借鉴 Claude Code 的 splitSysPromptPrefix）：
    1. 识别 DYNAMIC_BOUNDARY 标记
    2. 边界前的 section → global scope（可跨组织缓存）
    3. 边界后的 section → 不缓存（per-session 动态内容）
    4. 如果 skip_global_cache=True（如存在 MCP tools），全部降级为 org scope
    5. 如果 enable_caching=False，全部 cache_scope=None
    """
    if not enable_caching:
        return [SystemPromptBlock(text=system_prompt_to_str(prompt), cache_scope=None)]

    # 查找边界
    boundary_idx = None
    for i, s in enumerate(prompt):
        if s.strip() == DYNAMIC_BOUNDARY:
            boundary_idx = i
            break

    if boundary_idx is not None and not skip_global_cache:
        # 有边界 + 允许 global cache
        static_sections = [s for s in prompt[:boundary_idx] if s.strip() and s.strip() != DYNAMIC_BOUNDARY]
        dynamic_sections = [s for s in prompt[boundary_idx + 1:] if s.strip()]
        blocks: list[SystemPromptBlock] = []
        if static_sections:
            blocks.append(SystemPromptBlock(text="\n\n".join(static_sections), cache_scope="global"))
        if dynamic_sections:
            blocks.append(SystemPromptBlock(text="\n\n".join(dynamic_sections), cache_scope=None))
        return blocks if blocks else [SystemPromptBlock(text="", cache_scope=None)]

    # 无边界或 skip_global_cache → 全部 org scope
    text = system_prompt_to_str(prompt)
    return [SystemPromptBlock(text=text, cache_scope="org" if text else None)]


# ── Content Block 类型 ──


@dataclass(slots=True)
class ThinkingBlock:
    thinking: str
    type: Literal["thinking"] = "thinking"


@dataclass(slots=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


@dataclass(slots=True)
class Message:
    role: Role
    content: list[ContentBlock]
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def user(text: str) -> "Message":
        return Message(role="user", content=[TextBlock(text=text)])

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(role="assistant", content=[TextBlock(text=text)])
