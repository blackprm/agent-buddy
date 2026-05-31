from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelClient, ModelResponse, StreamDelta
from agent_core.model.retry import stream_with_retries
from agent_core.recovery import ensure_tool_result_pairing, recover_messages_for_resume
from agent_core.tools.base import ToolContext, ToolRegistry, ToolResult, execute_tools_serially
from agent_core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def test_conversation_repair_inserts_missing_tool_result() -> None:
    messages = [Message(role="assistant", content=[ToolUseBlock(id="toolu_1", name="bash", input={"command": "false"})])]

    repaired, report = ensure_tool_result_pairing(messages)

    assert report.repaired is True
    assert report.inserted_missing_results == ["toolu_1"]
    assert repaired[1].role == "user"
    result = repaired[1].content[0]
    assert isinstance(result, ToolResultBlock)
    assert result.is_error is True


def test_conversation_repair_strips_orphan_tool_result() -> None:
    messages = [Message(role="user", content=[ToolResultBlock(tool_use_id="missing", content="old", is_error=True)])]

    repaired, report = ensure_tool_result_pairing(messages)

    assert report.repaired is True
    assert report.removed_orphan_results == ["missing"]
    assert isinstance(repaired[0].content[0], TextBlock)


def test_resume_recovery_filters_blank_and_adds_sentinel() -> None:
    messages = [Message.user("continue"), Message(role="assistant", content=[TextBlock(text="   ")])]

    recovered, report = recover_messages_for_resume(messages)

    assert report.removed_blank_assistant == 1
    assert report.inserted_sentinel is True
    assert recovered[-1].role == "assistant"
    assert isinstance(recovered[-1].content[0], TextBlock)
    assert "No response" in recovered[-1].content[0].text


class ExplodingTool:
    name = "explode"
    description = "explode"
    input_schema = {"type": "object", "properties": {}}
    is_concurrency_safe = False
    should_defer = False

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        raise KeyError("path")


@pytest.mark.asyncio
async def test_tool_exception_is_classified_as_recoverable_validation_error() -> None:
    registry = ToolRegistry([ExplodingTool()])
    tool_use = ToolUseBlock(id="toolu_1", name="explode", input={"file": "x"})

    [(returned_use, result)] = await execute_tools_serially(
        [tool_use],
        registry,
        ToolContext(session_id="s", messages=[]),
    )

    assert returned_use is tool_use
    assert result.is_error is True
    assert result.metadata["failureCategory"] == "input_validation"
    assert "missing required input field" in result.content


class FlakyStreamModel(ModelClient):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, system, messages, tools, metadata=None) -> ModelResponse:
        return ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn")

    async def stream(self, *, system, messages, tools, metadata=None) -> AsyncIterator[StreamDelta]:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary 529 overloaded")
        yield StreamDelta(type="text_delta", text="ok")
        yield StreamDelta(type="stop", stop_reason="end_turn")


@pytest.mark.asyncio
async def test_model_retry_yields_retry_event_then_recovers() -> None:
    model = FlakyStreamModel()
    items = []
    async for item in stream_with_retries(model, system="", messages=[], tools=[]):
        items.append(item)

    assert model.calls == 2
    assert items[0].__class__.__name__ == "ModelRetryEvent"
    assert items[1].type == "text_delta"


@pytest.mark.asyncio
async def test_runtime_emits_api_retry_event() -> None:
    runtime = AgentRuntime(
        model=FlakyStreamModel(),
        tools=ToolRegistry([]),
        config=AgentRuntimeConfig(session_id="retry-test"),
    )

    events = [event async for event in runtime.run("hello")]

    assert any(event.type == "api_retry" for event in events)
    assert any(event.type == "loop_completed" for event in events)
