from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.tasks import FileTaskStore, TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool
from agent_core.hooks.engine import HookEngine
from agent_core.hooks.types import HookEvent, HookInput, HookResult
from agent_core.tools.base import ToolContext, ToolRegistry
from agent_core.tools.builtin import ToolSearchTool


def test_task_store_uses_numeric_ids_and_highwatermark(tmp_path: Path) -> None:
    store = FileTaskStore(tmp_path / "tasks", task_list_id="session/one")

    first = store.create(subject="Create schema", description="Define tables")
    second = store.create(subject="Write API", description="Add endpoints")

    assert first.id == "1"
    assert second.id == "2"
    assert (tmp_path / "tasks" / "session-one" / "1.json").exists()

    assert store.delete("2") is True
    third = store.create(subject="Write tests", description="Cover API")
    assert third.id == "3"


def test_task_store_blocks_and_claims_like_claude_code(tmp_path: Path) -> None:
    store = FileTaskStore(tmp_path / "tasks", task_list_id="s1")
    schema = store.create(subject="Create schema", description="Define tables")
    api = store.create(subject="Write API", description="Add endpoints")

    assert store.block_task(schema.id, api.id) is True
    assert store.get(schema.id).blocks == [api.id]
    assert store.get(api.id).blockedBy == [schema.id]

    blocked = store.claim(api.id, "agent-a")
    assert blocked.success is False
    assert blocked.reason == "blocked"
    assert blocked.blockedByTasks == [schema.id]

    claimed = store.claim(schema.id, "agent-a")
    assert claimed.success is True
    assert store.get(schema.id).owner == "agent-a"
    # Real Claude Code claim only sets owner; status is changed by TaskUpdate.
    assert store.get(schema.id).status == "pending"

    store.update(schema.id, status="completed")
    unblocked = store.claim(api.id, "agent-a", check_agent_busy=True)
    assert unblocked.success is True


def test_task_store_rejects_dependency_cycles(tmp_path: Path) -> None:
    store = FileTaskStore(tmp_path / "tasks", task_list_id="s1")
    one = store.create(subject="One", description="First")
    two = store.create(subject="Two", description="Second")

    assert store.block_task(one.id, two.id) is True
    with pytest.raises(ValueError, match="cycle"):
        store.block_task(two.id, one.id)


async def test_task_tools_create_update_list_get(tmp_path: Path) -> None:
    store = FileTaskStore(tmp_path / "tasks", task_list_id="unused")
    context = ToolContext(session_id="web-session", messages=[])

    create = TaskCreateTool(store)
    update = TaskUpdateTool(store)
    list_tool = TaskListTool(store)
    get = TaskGetTool(store)

    created = await create.call({"subject": "Create schema", "description": "Define tables"}, context)
    assert created.is_error is False
    assert "Task #1 created successfully" in created.content

    created_2 = await create.call({"subject": "Write API", "description": "Add endpoints"}, context)
    assert "Task #2 created successfully" in created_2.content

    dep = await update.call({"taskId": "2", "addBlockedBy": ["1"]}, context)
    assert dep.is_error is False
    assert "blockedBy" in dep.metadata["updatedFields"]

    listed = await list_tool.call({}, context)
    assert "#2 [pending] Write API [blocked by #1]" in listed.content

    details = await get.call({"taskId": "2"}, context)
    assert "Blocked by: #1" in details.content

    done = await update.call({"taskId": "1", "status": "completed"}, context)
    assert done.is_error is False
    listed_after = await list_tool.call({}, context)
    assert "#2 [pending] Write API" in listed_after.content
    assert "blocked by #1" not in listed_after.content


async def test_task_tools_fire_hooks_and_emit_task_state(tmp_path: Path) -> None:
    store = FileTaskStore(tmp_path / "tasks", task_list_id="unused")
    seen: list[HookEvent] = []
    events: list[str] = []
    engine = HookEngine()

    def record_hook(hook_input: HookInput) -> HookResult:
        seen.append(hook_input.event)
        return HookResult(hook=None)  # type: ignore[arg-type]

    async def emit(event) -> None:
        events.append(event.type)

    engine.register_callback(HookEvent.TaskCreated, record_hook)
    engine.register_callback(HookEvent.TaskCompleted, record_hook)
    context = ToolContext(session_id="web-session", messages=[], hook_engine=engine, event_callback=emit)

    create = TaskCreateTool(store)
    update = TaskUpdateTool(store)
    await create.call({"subject": "Create schema", "description": "Define tables"}, context)
    await update.call({"taskId": "1", "status": "completed"}, context)

    assert HookEvent.TaskCreated in seen
    assert HookEvent.TaskCompleted in seen
    assert "task_state" in events


async def test_tool_search_defers_and_activates_task_tools(tmp_path: Path) -> None:
    task_store = FileTaskStore(tmp_path / "tasks", task_list_id="unused")
    search = ToolSearchTool()
    registry = ToolRegistry([search, TaskCreateTool(task_store), TaskListTool(task_store)])
    search.bind(registry)

    visible_before = [schema["name"] for schema in registry.schemas()]
    assert "ToolSearch" in visible_before
    assert "TaskCreate" not in visible_before

    result = await search.call({"query": "task"}, ToolContext(session_id="s1", messages=[]))
    assert result.is_error is False
    assert "TaskCreate" in result.metadata["activated"]
    assert "TaskCreate" in [schema["name"] for schema in registry.schemas()]
