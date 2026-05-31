from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent_core.model.base import ModelClient, ModelResponse, StreamDelta
from agent_core.types import Message, SystemPrompt, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, system_prompt_to_str


class OpenAICompatibleModelClient(ModelClient):
    """OpenAI 兼容 API 适配器。

    适用于：
    - OpenAI (GPT-4o, GPT-4, etc.)
    - 火山方舟 / ARK (doubao)
    - DeepSeek
    - 任何 OpenAI 兼容的 API 网关

    配置方式（环境变量）：
        OPENAI_API_KEY     — API Key
        OPENAI_BASE_URL    — API 基地址（如 https://ark.cn-beijing.volces.com/api/v3）
        OPENAI_MODEL       — 模型 ID（如 ep-20260524094932-k6jx2 或 gpt-4o）

    也支持代码直接传参。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model
        self._max_tokens = max_tokens

    # ── complete ──

    async def complete(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        max_tokens = (metadata or {}).get("max_tokens", self._max_tokens)
        system_str = system if isinstance(system, str) else system_prompt_to_str(system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self._build_messages(system_str, messages),
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    # ── stream ──

    async def stream(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        max_tokens = (metadata or {}).get("max_tokens", self._max_tokens)
        system_str = system if isinstance(system, str) else system_prompt_to_str(system)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": self._build_messages(system_str, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        # OpenAI-compatible streaming can interleave multiple tool calls by
        # ``index``: chunk 1 may announce calls 0..N, later chunks append
        # arguments for each index independently.  Buffer them by index and
        # emit canonical sequential tool_use deltas at finish.  Do not re-emit
        # accumulated JSON when a new tool call starts: the AgentRuntime also
        # accumulates deltas, so doing both duplicates arguments and turns
        # valid JSON into e.g. {..}{..}, which then parses as {}.
        tool_call_buffers: dict[int, dict[str, str]] = {}
        stop_reason: str | None = None
        usage_data: dict[str, Any] = {}

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # reasoning / thinking content (DeepSeek, Ark/Doubao reasoning
            # models and several OpenAI-compatible gateways expose this as an
            # extra delta field such as reasoning_content). Normalize it into
            # thinking_delta so the WebSocket frontend can render it.
            reasoning_text = _extract_reasoning_text(delta)
            if reasoning_text:
                yield StreamDelta(type="thinking_delta", text=reasoning_text)

            # text content
            if delta.content:
                yield StreamDelta(type="text_delta", text=delta.content)

            # tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = int(getattr(tc, "index", 0) or 0)
                    buf = tool_call_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        buf["id"] = tc.id
                    function = getattr(tc, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            buf["name"] = name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            buf["arguments"] += arguments

            # finish
            if choice.finish_reason is not None:
                # flush buffered tool_use blocks in index order
                for index in sorted(tool_call_buffers):
                    buf = tool_call_buffers[index]
                    tool_id = buf.get("id", "")
                    tool_name = buf.get("name", "")
                    if not tool_id or not tool_name:
                        continue
                    yield StreamDelta(
                        type="tool_use_start",
                        tool_use_id=tool_id,
                        tool_use_name=tool_name,
                    )
                    yield StreamDelta(
                        type="tool_use_delta",
                        tool_use_id=tool_id,
                        tool_use_input_delta=buf.get("arguments", ""),
                    )
                stop_reason = self._map_stop_reason(choice.finish_reason)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }
        yield StreamDelta(type="stop", stop_reason=stop_reason, usage=usage_data)

    # ── helpers ──

    def _build_messages(self, system: str, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "system":
                continue
            # 如果 content 只有单个 TextBlock，用简单格式
            if len(msg.content) == 1 and isinstance(msg.content[0], TextBlock):
                result.append({"role": msg.role, "content": msg.content[0].text})
                continue
            # 多 block / tool_use / tool_result
            parts: list[dict[str, Any]] = []
            text_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    # OpenAI 没有 thinking block，跳过或放注释
                    pass
                elif isinstance(block, ToolUseBlock):
                    if text_parts:
                        parts.append({"type": "text", "text": "\n".join(text_parts)})
                        text_parts = []
                    parts.append({
                        "type": "function",
                        "id": block.id,
                        "function": {"name": block.name, "arguments": json.dumps(block.input)},
                    })
                elif isinstance(block, ToolResultBlock):
                    if text_parts:
                        parts.append({"type": "text", "text": "\n".join(text_parts)})
                        text_parts = []
                    parts.append({
                        "type": "function",
                        "id": block.tool_use_id,
                        "function": {
                            "name": "tool_result",
                            "arguments": json.dumps({"content": block.content, "is_error": block.is_error}),
                        },
                    })
            if text_parts:
                parts.append({"type": "text", "text": "\n".join(text_parts)})

            # 如果只有 text parts，用简单格式
            if all(p.get("type") == "text" for p in parts):
                result.append({"role": msg.role, "content": "\n".join(p["text"] for p in parts)})
            else:
                # 有 tool_use 时用 assistant + tool_calls 格式
                if msg.role == "assistant":
                    tool_calls = [p for p in parts if p.get("type") == "function"]
                    text_content = "\n".join(p["text"] for p in parts if p.get("type") == "text") or None
                    entry: dict[str, Any] = {"role": "assistant"}
                    if text_content:
                        entry["content"] = text_content
                    if tool_calls:
                        entry["tool_calls"] = tool_calls
                    result.append(entry)
                elif msg.role == "user":
                    # tool_result 需要用 tool message 格式
                    for p in parts:
                        if p.get("type") == "function" and p["function"]["name"] == "tool_result":
                            args = json.loads(p["function"]["arguments"])
                            result.append({
                                "role": "tool",
                                "tool_call_id": p["id"],
                                "content": args.get("content", ""),
                            })
                        elif p.get("type") == "text":
                            result.append({"role": "user", "content": p["text"]})
        return result

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把 Anthropic 格式的 tool schema 转成 OpenAI function calling 格式。"""
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            })
        return result

    def _parse_response(self, response: Any) -> ModelResponse:
        content_blocks: list[TextBlock | ToolUseBlock | ThinkingBlock] = []
        choice = response.choices[0] if response.choices else None
        if choice:
            msg = choice.message
            reasoning_text = _extract_reasoning_text(msg)
            if reasoning_text:
                content_blocks.append(ThinkingBlock(thinking=reasoning_text))
            if msg.content:
                content_blocks.append(TextBlock(text=msg.content))
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}
                    content_blocks.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=tool_input))

        stop_reason = self._map_stop_reason(choice.finish_reason) if choice else "end_turn"
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return ModelResponse(content=content_blocks, stop_reason=stop_reason, usage=usage, raw=response)

    @staticmethod
    def _map_stop_reason(reason: str | None) -> str:
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_output_tokens",
            "content_filter": "end_turn",
        }
        return mapping.get(reason or "", "end_turn")


def _extract_reasoning_text(obj: Any) -> str:
    """Extract reasoning/thinking text from OpenAI-compatible responses."""
    for key in ("reasoning_content", "reasoning", "reasoning_text", "thinking", "thought"):
        value = _get_extra_attr(obj, key)
        if isinstance(value, str) and value:
            return value

    details = _get_extra_attr(obj, "reasoning_details") or _get_extra_attr(obj, "reasoning_details_delta")
    if isinstance(details, list):
        parts: list[str] = []
        for item in details:
            for key in ("text", "content", "reasoning", "reasoning_content"):
                value = _get_extra_attr(item, key)
                if isinstance(value, str) and value:
                    parts.append(value)
                    break
        return "".join(parts)
    return ""


def _get_extra_attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    value = getattr(obj, key, None)
    if value is not None:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)
    return None
