from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from agent_core.types import ContentBlock, Message, SystemPrompt, system_prompt_to_str


@dataclass(slots=True)
class ModelResponse:
    content: list[ContentBlock]
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None


@dataclass(slots=True)
class StreamDelta:
    """A single chunk from a streaming model response."""
    type: str  # "text_delta" | "thinking_delta" | "tool_use_start" | "tool_use_delta" | "stop"
    text: str = ""
    tool_use_id: str = ""
    tool_use_name: str = ""
    tool_use_input_delta: str = ""
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    """模型适配器协议。

    Web 后端里建议把真实 provider 隔离在这里：AgentRuntime 不知道 Anthropic、OpenAI、
    火山、LiteLLM 或内部网关的具体 SDK。
    """

    async def complete(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        ...

    async def stream(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """流式调用模型，逐 chunk 返回 StreamDelta。

        默认实现回退到 complete() 一次性返回。支持流式的适配器应覆盖此方法。
        """
        # 默认回退：一次性调用 complete，把结果拆成 delta 序列
        response = await self.complete(system=system, messages=messages, tools=tools, metadata=metadata)
        for block in response.content:
            if hasattr(block, "thinking") and block.thinking:  # ThinkingBlock
                yield StreamDelta(type="thinking_delta", text=block.thinking)
            elif hasattr(block, "text") and block.text:  # TextBlock
                yield StreamDelta(type="text_delta", text=block.text)
            elif hasattr(block, "id") and hasattr(block, "name"):  # ToolUseBlock
                import json
                yield StreamDelta(
                    type="tool_use_start",
                    tool_use_id=block.id,
                    tool_use_name=block.name,
                )
                yield StreamDelta(
                    type="tool_use_delta",
                    tool_use_id=block.id,
                    tool_use_input_delta=json.dumps(block.input),
                )
        yield StreamDelta(type="stop", stop_reason=response.stop_reason, usage=response.usage)
