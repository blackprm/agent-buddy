from __future__ import annotations

import asyncio

import pytest

from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig, USER_INTERRUPT_MESSAGE
from agent_core.core.events import AgentEvent
from agent_core.model.base import ModelResponse, StreamDelta
from agent_core.tools.base import ToolRegistry
from agent_core.types import Message, TextBlock
from agent_server.task_manager import AgentTaskManager


class FakePermissionRuntime:
    async def run(self, _message: str, abort_event=None):
        yield AgentEvent("permission_request", {
            "tool_use_id": "toolu_test",
            "tool": "bash",
            "input": {"command": "pwd"},
            "reason": "Bash requires permission",
            "options": [{"type": "accept-once", "label": "Allow once"}, {"type": "reject", "label": "Deny"}],
            "metadata": {},
        })
        while not (abort_event and abort_event.is_set()):
            await asyncio.sleep(0.01)


class FakeIdleRuntime:
    async def run(self, _message: str, abort_event=None):
        while not (abort_event and abort_event.is_set()):
            await asyncio.sleep(0.01)
            if False:
                yield None


class FakeCancellableRuntime:
    def __init__(self) -> None:
        self.cancelled = False
        self.cancellations: list[dict[str, str]] = []

    def record_user_cancellation(self, *, reason: str = "", phase: str = "", persist: bool = True) -> bool:
        self.cancellations.append({"reason": reason, "phase": phase, "persist": str(persist)})
        return True

    async def run(self, _message: str, abort_event=None):
        try:
            yield AgentEvent("tool_started", {"tool_use_id": "toolu_sleep", "tool": "SleepTool"})
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class RecordingModelClient:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(self, *, system, messages: list[Message], tools, metadata=None):
        self.calls.append(list(messages))
        return ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn")

    async def stream(self, *, system, messages: list[Message], tools, metadata=None):
        self.calls.append(list(messages))
        yield StreamDelta(type="text_delta", text="ok")
        yield StreamDelta(type="stop", stop_reason="end_turn", usage={})


@pytest.mark.asyncio
async def test_task_manager_exposes_pending_permission_for_reconnect_replay() -> None:
    manager = AgentTaskManager()
    session_id = "pending-permission-replay-test"
    managed = await manager.start(session_id=session_id, runtime=FakePermissionRuntime(), message="run pwd")  # type: ignore[arg-type]

    assert managed is not None
    assert await manager.wait_for_next(session_id, 0) is True

    pending = await manager.pending_permission(session_id)
    assert pending is not None
    assert pending["tool_use_id"] == "toolu_test"
    assert pending["tool"] == "bash"

    assert await manager.respond_permission(session_id, "allow", {"type": "accept-once"}) is True
    assert await manager.pending_permission(session_id) is None

    await manager.shutdown()


@pytest.mark.asyncio
async def test_agent_runtime_returns_user_interrupt_message_to_next_model_turn() -> None:
    model = RecordingModelClient()
    runtime = AgentRuntime(
        model=model,  # type: ignore[arg-type]
        tools=ToolRegistry([]),
        context_builder=ContextBuilder(base_instructions="test"),
        config=AgentRuntimeConfig(session_id="interrupt-visible-test", session_memory_enabled=False),
    )
    abort_event = asyncio.Event()
    abort_event.set()

    first_events = [event async for event in runtime.run("run something long", abort_event=abort_event)]

    assert any(event.type == "loop_aborted" for event in first_events)
    assert runtime.messages[-1].metadata["user_interrupt"] is True
    assert runtime.messages[-1].content[0].text == USER_INTERRUPT_MESSAGE  # type: ignore[attr-defined]

    second_events = [event async for event in runtime.run("what happened?")]

    assert any(event.type == "loop_completed" for event in second_events)
    assert model.calls
    visible_text = "\n".join(
        block.text
        for message in model.calls[-1]
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert USER_INTERRUPT_MESSAGE in visible_text


@pytest.mark.asyncio
async def test_task_manager_ignores_stale_permission_response_without_pending_request() -> None:
    manager = AgentTaskManager()
    session_id = "stale-permission-response-test"
    managed = await manager.start(session_id=session_id, runtime=FakeIdleRuntime(), message="idle")  # type: ignore[arg-type]

    assert await manager.respond_permission("missing-session", "allow", {"type": "accept-once"}) is False
    assert managed is not None
    assert await manager.respond_permission(session_id, "allow", {"type": "accept-once"}) is False

    await manager.shutdown()


@pytest.mark.asyncio
async def test_task_manager_abort_cancels_running_runtime_task() -> None:
    manager = AgentTaskManager()
    session_id = "cancel-running-tool-test"
    runtime = FakeCancellableRuntime()
    managed = await manager.start(session_id=session_id, runtime=runtime, message="run slow tool")  # type: ignore[arg-type]

    assert managed is not None
    assert await manager.wait_for_next(session_id, 0) is True
    assert await manager.abort(session_id) is True
    for _ in range(20):
        await manager.wait_for_next(session_id, 1)
        if managed.done:
            break

    events, _, done = await manager.events_after(session_id, 0)
    assert done is True
    assert runtime.cancelled is True
    assert runtime.cancellations
    assert runtime.cancellations[0]["reason"] == "external_abort"
    assert runtime.cancellations[0]["phase"] == "task_manager_abort"
    assert any(event and event.type == "loop_aborted" for _, event in events)

    await manager.shutdown()
