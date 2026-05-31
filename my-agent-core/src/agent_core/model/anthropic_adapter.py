from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent_core.model.base import ModelClient, ModelResponse, StreamDelta
from agent_core.types import (
    Message,
    SystemPrompt,
    SystemPromptBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    system_prompt_to_str,
    system_prompt_to_blocks,
)


class AnthropicMessagesModelClient(ModelClient):
    """Anthropic Messages API 适配器。

    依赖可选安装：`pip install -e '.[anthropic]'`。
    这里故意只放 provider 映射，不把 SDK 类型泄露到 AgentRuntime。
    """

    def __init__(self, *, model: str, api_key: str | None = None, base_url: str | None = None, enable_caching: bool = False) -> None:
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._enable_caching = enable_caching

    async def complete(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        system_param = self._build_system_param(system)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=(metadata or {}).get("max_tokens", 4096),
            system=system_param,
            messages=[self._to_anthropic_message(message) for message in messages if message.role != "system"],
            tools=tools,
        )
        return ModelResponse(
            content=[self._from_anthropic_block(block) for block in response.content],
            stop_reason=response.stop_reason,
            usage=response.usage.model_dump() if hasattr(response.usage, "model_dump") else {},
            raw=response,
        )

    async def stream(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """流式调用 Anthropic Messages API。"""
        system_param = self._build_system_param(system)
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=(metadata or {}).get("max_tokens", 4096),
            system=system_param,
            messages=[self._to_anthropic_message(message) for message in messages if message.role != "system"],
            tools=tools,
        ) as stream:
            usage_data: dict[str, Any] = {}
            async for event in stream:
                if event.type == "message_start":
                    usage = getattr(event.message, "usage", None)
                    if usage:
                        usage_data.update(usage.model_dump() if hasattr(usage, "model_dump") else {})
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "thinking":
                        yield StreamDelta(type="thinking_delta", text="")
                    elif block.type == "text":
                        yield StreamDelta(type="text_delta", text="")
                    elif block.type == "tool_use":
                        yield StreamDelta(
                            type="tool_use_start",
                            tool_use_id=block.id,
                            tool_use_name=block.name,
                        )
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        yield StreamDelta(type="thinking_delta", text=delta.thinking)
                    elif delta.type == "text_delta":
                        yield StreamDelta(type="text_delta", text=delta.text)
                    elif delta.type == "input_json_delta":
                        yield StreamDelta(
                            type="tool_use_delta",
                            tool_use_id="",
                            tool_use_input_delta=delta.partial_json,
                        )
                elif event.type == "message_delta":
                    usage = event.usage
                    if usage:
                        usage_data.update(usage.model_dump() if hasattr(usage, "model_dump") else {"output_tokens": usage.output_tokens})
                    yield StreamDelta(
                        type="stop",
                        stop_reason=event.delta.stop_reason,
                        usage=usage_data,
                    )

    def _to_anthropic_message(self, message: Message) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                content.append({"type": "thinking", "thinking": block.thinking})
            elif isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif isinstance(block, ToolResultBlock):
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        return {"role": message.role, "content": content}

    def _from_anthropic_block(self, block: Any) -> TextBlock | ToolUseBlock | ThinkingBlock:
        if block.type == "thinking":
            return ThinkingBlock(thinking=block.thinking)
        if block.type == "text":
            return TextBlock(text=block.text)
        if block.type == "tool_use":
            return ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {}))
        return TextBlock(text=str(block))

    def _build_system_param(self, system: str | SystemPrompt) -> str | list[dict[str, Any]]:
        """将 system 参数转换为 Anthropic SDK 格式。

        - str → 直接传递（简单模式）
        - SystemPrompt → 根据 enable_caching 决定是否拆分为带 cache_control 的 block 列表

        Anthropic SDK 的 system 参数支持：
        - str: 单个 system prompt
        - list[dict]: 多个 system prompt block，可带 cache_control
          [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral", "scope": "global"}}]
        """
        if isinstance(system, str):
            return system

        if not self._enable_caching:
            return system_prompt_to_str(system)

        # 启用缓存：拆分为带 cache_control 的 block 列表
        blocks = system_prompt_to_blocks(system, enable_caching=True)
        result: list[dict[str, Any]] = []
        for block in blocks:
            if not block.text:
                continue
            entry: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cache_scope is not None:
                cache_control: dict[str, Any] = {"type": "ephemeral"}
                if block.cache_scope == "global":
                    cache_control["scope"] = "global"
                entry["cache_control"] = cache_control
            result.append(entry)
        return result if result else system_prompt_to_str(system)
