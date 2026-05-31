from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Awaitable

from agent_core.types import (
    DYNAMIC_BOUNDARY,
    SystemPrompt,
    SystemPromptBlock,
    as_system_prompt,
    system_prompt_to_str,
    system_prompt_to_blocks,
)


# ── Section 定义 ──────────────────────────────────────────────


@dataclass(slots=True)
class SystemPromptSection:
    """命名 section：可缓存计算结果，支持同步/异步 compute。

    借鉴 Claude Code 的 systemPromptSection / DANGEROUS_uncachedSystemPromptSection：
    - cache_break=False: 计算一次后缓存，直到 clear_cache() 被调用
    - cache_break=True: 每轮重新计算（用于 MCP instructions 等动态内容）
    """
    name: str
    compute: Callable[[], str | None] | Callable[[], Awaitable[str | None]]
    cache_break: bool = False


def system_prompt_section(
    name: str,
    compute: Callable[[], str | None] | Callable[[], Awaitable[str | None]],
) -> SystemPromptSection:
    """创建可缓存的 section（默认行为）。"""
    return SystemPromptSection(name=name, compute=compute, cache_break=False)


def uncached_system_prompt_section(
    name: str,
    compute: Callable[[], str | None] | Callable[[], Awaitable[str | None]],
    reason: str = "",
) -> SystemPromptSection:
    """创建不缓存的 section（每轮重新计算）。

    用于 MCP instructions 等可能在 turn 之间变化的内容。
    reason 参数仅作文档用途，不影响逻辑。
    """
    return SystemPromptSection(name=name, compute=compute, cache_break=True)


# ── ContextBuilder ────────────────────────────────────────────


@dataclass(slots=True)
class ContextBuilder:
    """Section 化 system prompt 构建器。

    借鉴 Claude Code 的 system prompt 管理架构：
    - Section 列表代替单字符串，支持细粒度缓存控制
    - 静态/动态边界（DYNAMIC_BOUNDARY）分隔可缓存和不可缓存内容
    - 优先级链：override > agent > custom > default
    - prepend_user_context / append_system_context 注入运行时上下文
    - Section 计算缓存，避免每轮重复计算静态内容
    """

    product_name: str = "MyAgent"
    base_instructions: str = "You are a helpful autonomous agent. Use tools when needed."
    extra_sections: list[str] = field(default_factory=list)

    # ── 优先级链 ──
    override_prompt: str | None = None
    agent_prompt: str | None = None
    custom_prompt: str | None = None
    append_prompt: str | None = None

    # ── 外部注入的静态 section（替代 _default_sections 的硬编码输出）──
    static_sections: list[str] | None = None

    # ── 动态 section 注册表 ──
    dynamic_sections: list[SystemPromptSection] = field(default_factory=list)

    # ── Section 计算缓存 ──
    _section_cache: dict[str, str | None] = field(default_factory=dict, repr=False)

    # ── 运行时上下文注入 ──
    _prepend_context: dict[str, str] = field(default_factory=dict, repr=False)
    _append_context: dict[str, str] = field(default_factory=dict, repr=False)

    def clear_cache(self) -> None:
        """清空 section 计算缓存（类似 Claude Code 的 /clear 行为）。"""
        self._section_cache.clear()

    def prepend_user_context(self, context: dict[str, str]) -> None:
        """注入用户上下文到 system prompt 动态区域开头。

        借鉴 Claude Code 的 prependUserContext，但注入到 system prompt
        而非 user messages（因为我们的通用底座不一定有 user message 注入点）。
        """
        self._prepend_context.update(context)

    def append_system_context(self, context: dict[str, str]) -> None:
        """追加系统上下文到 system prompt 末尾。

        借鉴 Claude Code 的 appendSystemContext。
        """
        self._append_context.update(context)

    async def build(self) -> SystemPrompt:
        """构建完整的 system prompt（异步，因为 section compute 可能是异步的）。

        返回 SystemPrompt（list[str] 品牌类型），包含：
        1. 优先级链选择的基础 prompt
        2. DYNAMIC_BOUNDARY 边界标记
        3. 动态 section（带缓存）
        4. prepend/append 注入的上下文
        """
        # ── 1. 优先级链选择基础 prompt ──
        base_sections = self._resolve_priority_chain()

        # ── 2. 静态 section（边界之前）──
        sections: list[str] = list(base_sections)

        # ── 3. 动态边界 ──
        sections.append(DYNAMIC_BOUNDARY)

        # ── 4. prepend 上下文 ──
        if self._prepend_context:
            lines = [f"{k}: {v}" for k, v in self._prepend_context.items()]
            sections.append("\n".join(lines))

        # ── 5. 动态 section（带缓存）──
        for sec in self.dynamic_sections:
            value = await self._resolve_section(sec)
            if value:
                sections.append(value)

        # ── 6. append 上下文 ──
        if self._append_context:
            lines = [f"{k}: {v}" for k, v in self._append_context.items()]
            sections.append("\n".join(lines))

        return as_system_prompt(sections)

    def build_str(self) -> str:
        """同步构建 system prompt 字符串（用于简单场景，不解析动态 section）。

        注意：此方法不执行动态 section 的 compute，仅返回静态内容。
        完整构建请使用 async build()。
        """
        sections = self._resolve_priority_chain()
        return "\n\n".join(s for s in sections if s.strip())

    def _resolve_priority_chain(self) -> list[str]:
        """优先级链：override > agent > custom > default。

        借鉴 Claude Code 的 buildEffectiveSystemPrompt：
        - override: 完全替换（如 loop mode）
        - agent: agent 自定义 prompt（追加或替换，取决于模式）
        - custom: 用户自定义 prompt
        - default: 默认 prompt（product_name + base_instructions + 规则）
        - append: 始终追加到末尾
        """
        # 优先级 0: override — 完全替换
        if self.override_prompt:
            return [self.override_prompt]

        # 优先级 1-3: agent / custom / default
        if self.agent_prompt:
            # agent prompt 替换 default
            base = [self.agent_prompt]
        elif self.custom_prompt:
            base = [self.custom_prompt]
        else:
            base = self._default_sections()

        # 追加 append_prompt
        if self.append_prompt:
            base.append(self.append_prompt)

        return base

    def _default_sections(self) -> list[str]:
        """默认 system prompt sections。"""
        # 如果 static_sections 已设置，直接使用（由 PromptStore 注入）
        if self.static_sections is not None:
            return [s for s in self.static_sections if s.strip()]

        sections = [
            f"# {self.product_name}",
            self.base_instructions,
            "# Runtime Context",
            f"Current date: {date.today().isoformat()}",
            "# Tool Use Rules",
            "When you need external information or side effects, call an available tool. "
            "After tool results are returned, continue reasoning from the results.",
        ]
        sections.extend(self.extra_sections)
        return [s for s in sections if s.strip()]

    async def _resolve_section(self, section: SystemPromptSection) -> str | None:
        """解析单个 section，带缓存。"""
        if not section.cache_break and section.name in self._section_cache:
            return self._section_cache[section.name]

        result = section.compute()
        if inspect.isawaitable(result):
            result = await result

        self._section_cache[section.name] = result
        return result
