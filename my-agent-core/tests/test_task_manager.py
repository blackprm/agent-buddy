from __future__ import annotations

import asyncio

import pytest

from agent_core.core.events import AgentEvent
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
async def test_task_manager_ignores_stale_permission_response_without_pending_request() -> None:
    manager = AgentTaskManager()
    session_id = "stale-permission-response-test"
    managed = await manager.start(session_id=session_id, runtime=FakeIdleRuntime(), message="idle")  # type: ignore[arg-type]

    assert await manager.respond_permission("missing-session", "allow", {"type": "accept-once"}) is False
    assert managed is not None
    assert await manager.respond_permission(session_id, "allow", {"type": "accept-once"}) is False

    await manager.shutdown()
