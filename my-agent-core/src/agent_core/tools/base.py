from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_core.types import Message, ToolUseBlock
from agent_core.recovery.tool_errors import format_tool_exception


@dataclass(slots=True)
class ToolContext:
    session_id: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)
    cwd: str = ""
    ask_callback: Any | None = None
    hook_engine: Any | None = None
    event_callback: Any | None = None

    def resolve_path(self, raw_path: str | Path | None, *, default: str | Path = ".") -> Path:
        """Resolve tool paths relative to the runtime workspace, not server cwd."""
        path = Path(raw_path if raw_path not in (None, "") else default).expanduser()
        if not path.is_absolute():
            path = Path(self.cwd or ".") / path
        return path.resolve()


@dataclass(slots=True)
class ToolResult:
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    is_concurrency_safe: bool
    should_defer: bool

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        ...


class ToolRegistry:
    """Registry 模式：统一管理工具定义与分发。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._activated_deferred: set[str] = set()
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
            if not getattr(tool, "should_defer", False) or tool.name in self._activated_deferred
        ]

    def all_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "deferred": bool(getattr(tool, "should_defer", False)),
                "activated": tool.name in self._activated_deferred,
            }
            for tool in self._tools.values()
        ]

    def deferred_schemas(self) -> list[dict[str, Any]]:
        return [schema for schema in self.all_schemas() if schema.get("deferred")]

    def activate_deferred(self, names: list[str] | None = None) -> list[str]:
        activated: list[str] = []
        wanted = set(names or [])
        for name, tool in self._tools.items():
            if not getattr(tool, "should_defer", False):
                continue
            if wanted and name not in wanted:
                continue
            self._activated_deferred.add(name)
            activated.append(name)
        return activated


def _format_tool_exception(tool_use: ToolUseBlock, exc: Exception) -> str:
    """把工具异常格式化为模型可恢复的错误，而不是裸露 `'path'` 这类弱信息。"""
    return format_tool_exception(tool_use, exc).message


# ── Tool orchestration: partitionToolCalls + concurrent/serial execution ──


@dataclass(slots=True)
class ToolBatch:
    """A batch of tool calls that share the same concurrency safety."""
    is_concurrency_safe: bool
    blocks: list[ToolUseBlock] = field(default_factory=list)


def partition_tool_calls(
    tool_uses: list[ToolUseBlock],
    registry: ToolRegistry,
) -> list[ToolBatch]:
    """Partition tool calls into batches based on is_concurrency_safe.

    Consecutive concurrency-safe tools are grouped into one batch for parallel execution.
    Non-safe tools each get their own batch for serial execution.

    Mirrors Claude Code's partitionToolCalls in toolOrchestration.ts.
    """
    batches: list[ToolBatch] = []
    for tool_use in tool_uses:
        tool = registry.get(tool_use.name)
        is_safe = tool.is_concurrency_safe if tool else False
        if is_safe and batches and batches[-1].is_concurrency_safe:
            batches[-1].blocks.append(tool_use)
        else:
            batches.append(ToolBatch(is_concurrency_safe=is_safe, blocks=[tool_use]))
    return batches


async def execute_tools_concurrently(
    tool_uses: list[ToolUseBlock],
    registry: ToolRegistry,
    context: ToolContext,
    permission_check: Any | None = None,
) -> list[tuple[ToolUseBlock, ToolResult]]:
    """Execute a batch of concurrency-safe tools in parallel.

    Returns list of (ToolUseBlock, ToolResult) pairs in original order.
    """
    async def _run_one(tool_use: ToolUseBlock) -> tuple[ToolUseBlock, ToolResult]:
        tool = registry.get(tool_use.name)
        if tool is None:
            return tool_use, ToolResult(content=f"Unknown tool: {tool_use.name}", is_error=True)
        try:
            result = await tool.call(tool_use.input, context)
            return tool_use, result
        except Exception as exc:
            failure = format_tool_exception(tool_use, exc)
            return tool_use, ToolResult(content=failure.message, is_error=True, metadata=failure.to_metadata())

    results = await asyncio.gather(*[_run_one(tu) for tu in tool_uses])
    return list(results)


async def execute_tools_serially(
    tool_uses: list[ToolUseBlock],
    registry: ToolRegistry,
    context: ToolContext,
) -> list[tuple[ToolUseBlock, ToolResult]]:
    """Execute a batch of non-concurrency-safe tools one by one.

    Returns list of (ToolUseBlock, ToolResult) pairs in order.
    """
    results: list[tuple[ToolUseBlock, ToolResult]] = []
    for tool_use in tool_uses:
        tool = registry.get(tool_use.name)
        if tool is None:
            results.append((tool_use, ToolResult(content=f"Unknown tool: {tool_use.name}", is_error=True)))
            continue
        try:
            result = await tool.call(tool_use.input, context)
            results.append((tool_use, result))
        except Exception as exc:
            failure = format_tool_exception(tool_use, exc)
            results.append((tool_use, ToolResult(content=failure.message, is_error=True, metadata=failure.to_metadata())))
    return results
