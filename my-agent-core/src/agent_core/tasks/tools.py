from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.core.events import AgentEvent
from agent_core.tasks.store import FileTaskStore, TaskRecord, TaskStatus
from agent_core.tools.base import ToolContext, ToolResult


TASK_CREATE_PROMPT = """Use this tool to create a structured task list for complex, multi-step work.

Create tasks with clear imperative subjects and detailed descriptions. New
tasks are always pending and unowned. After creating tasks, use TaskUpdate to
set dependencies with addBlocks/addBlockedBy when needed. Check TaskList first
to avoid creating duplicates.
"""

TASK_UPDATE_PROMPT = """Use this tool to update a task in the task list.

Status workflow: pending -> in_progress -> completed. Use status=deleted to
remove a task. Only mark a task completed when the work is fully accomplished;
if tests fail or implementation is partial, keep it in_progress and create a
new blocking task if needed. Read latest state with TaskGet before updating
stale tasks.
"""


class _TaskToolBase:
    is_concurrency_safe = True
    should_defer = True

    def __init__(self, store: FileTaskStore | None = None, *, base_dir: str | Path | None = None) -> None:
        self._store = store or FileTaskStore(base_dir)

    def _store_for_context(self, context: ToolContext) -> FileTaskStore:
        explicit = context.metadata.get("task_list_id") or context.metadata.get("team_name")
        task_list_id = str(explicit or context.session_id or self._store.task_list_id)
        return self._store.for_task_list(task_list_id)

    def _result(self, content: str, **metadata: Any) -> ToolResult:
        return ToolResult(content=content, metadata=metadata)

    async def _emit_task_state(self, context: ToolContext, store: FileTaskStore) -> None:
        if not context.event_callback:
            return
        tasks = [task.to_dict() for task in store.list(include_internal=False)]
        await context.event_callback(AgentEvent("task_state", {"task_list_id": store.task_list_id, "tasks": tasks}))


class TaskCreateTool(_TaskToolBase):
    name = "TaskCreate"
    description = "Create a new persistent task in the task list.\n\n" + TASK_CREATE_PROMPT
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "A brief title for the task"},
            "description": {"type": "string", "description": "What needs to be done"},
            "activeForm": {"type": "string", "description": "Present continuous form shown when in_progress"},
            "metadata": {"type": "object", "description": "Arbitrary metadata to attach to the task"},
        },
        "required": ["subject", "description"],
    }

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store = self._store_for_context(context)
        try:
            task = store.create(
                subject=str(tool_input.get("subject") or ""),
                description=str(tool_input.get("description") or ""),
                active_form=tool_input.get("activeForm"),
                metadata=tool_input.get("metadata") if isinstance(tool_input.get("metadata"), dict) else None,
            )
            if context.hook_engine and hasattr(context.hook_engine, "fire_task_created"):
                hook_result = await context.hook_engine.fire_task_created(task=task.to_dict(), session_id=context.session_id)
                if hook_result.should_block:
                    store.delete(task.id)
                    return ToolResult(content=hook_result.stop_reason or "TaskCreate blocked by hook", is_error=True)
            await self._emit_task_state(context, store)
        except Exception as exc:
            return ToolResult(content=f"TaskCreate failed: {exc}", is_error=True)
        return self._result(
            f"Task #{task.id} created successfully: {task.subject}",
            task=task.to_dict(),
            taskListId=store.task_list_id,
            tasksDir=str(store.tasks_dir),
        )


class TaskListTool(_TaskToolBase):
    name = "TaskList"
    description = "List all persistent tasks with status, owner, and unresolved blockers."
    input_schema = {"type": "object", "properties": {}}

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store = self._store_for_context(context)
        tasks = store.list(include_internal=False)
        if not tasks:
            return self._result("No tasks found", tasks=[], taskListId=store.task_list_id)
        completed = {task.id for task in tasks if task.status == "completed"}
        lines: list[str] = []
        payload: list[dict[str, Any]] = []
        for task in tasks:
            unresolved_blockers = [task_id for task_id in task.blockedBy if task_id not in completed]
            owner = f" ({task.owner})" if task.owner else ""
            blocked = f" [blocked by {', '.join('#' + task_id for task_id in unresolved_blockers)}]" if unresolved_blockers else ""
            lines.append(f"#{task.id} [{task.status}] {task.subject}{owner}{blocked}")
            payload.append({
                "id": task.id,
                "subject": task.subject,
                "status": task.status,
                "owner": task.owner,
                "blockedBy": unresolved_blockers,
            })
        return self._result("\n".join(lines), tasks=payload, taskListId=store.task_list_id)


class TaskGetTool(_TaskToolBase):
    name = "TaskGet"
    description = "Retrieve full details of a task by ID."
    input_schema = {
        "type": "object",
        "properties": {"taskId": {"type": "string", "description": "The ID of the task to retrieve"}},
        "required": ["taskId"],
    }

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store = self._store_for_context(context)
        task = store.get(str(tool_input.get("taskId") or tool_input.get("task_id") or ""))
        if not task:
            return self._result("Task not found", task=None, taskListId=store.task_list_id)
        lines = [
            f"Task #{task.id}: {task.subject}",
            f"Status: {task.status}",
            f"Description: {task.description}",
        ]
        if task.owner:
            lines.append(f"Owner: {task.owner}")
        if task.blockedBy:
            lines.append(f"Blocked by: {', '.join('#' + task_id for task_id in task.blockedBy)}")
        if task.blocks:
            lines.append(f"Blocks: {', '.join('#' + task_id for task_id in task.blocks)}")
        return self._result("\n".join(lines), task=task.to_dict(), taskListId=store.task_list_id)


class TaskUpdateTool(_TaskToolBase):
    name = "TaskUpdate"
    description = "Update a task in the task list.\n\n" + TASK_UPDATE_PROMPT
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "The ID of the task to update"},
            "subject": {"type": "string", "description": "New subject for the task"},
            "description": {"type": "string", "description": "New description for the task"},
            "activeForm": {"type": "string", "description": "Present continuous form shown when in_progress"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"], "description": "New status"},
            "addBlocks": {"type": "array", "items": {"type": "string"}, "description": "Task IDs that this task blocks"},
            "addBlockedBy": {"type": "array", "items": {"type": "string"}, "description": "Task IDs that block this task"},
            "owner": {"type": "string", "description": "New owner for the task"},
            "metadata": {"type": "object", "description": "Metadata keys to merge; null deletes a key"},
        },
        "required": ["taskId"],
    }

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store = self._store_for_context(context)
        task_id = str(tool_input.get("taskId") or tool_input.get("task_id") or "")
        existing = store.get(task_id)
        if not existing:
            return self._result(f"Task #{task_id} not found", success=False, taskId=task_id, updatedFields=[], error="Task not found")

        status = tool_input.get("status")
        if status == "deleted":
            deleted = store.delete(task_id)
            await self._emit_task_state(context, store)
            return self._result(
                f"Updated task #{existing.id} deleted" if deleted else f"Failed to delete task #{existing.id}",
                success=deleted,
                taskId=existing.id,
                updatedFields=["deleted"] if deleted else [],
                statusChange={"from": existing.status, "to": "deleted"} if deleted else None,
            )

        updates: dict[str, Any] = {}
        updated_fields: list[str] = []
        for field, key in (("subject", "subject"), ("description", "description"), ("activeForm", "active_form"), ("owner", "owner")):
            if field in tool_input and tool_input.get(field) != getattr(existing, field if field != "activeForm" else "activeForm"):
                updates[key] = tool_input.get(field)
                updated_fields.append(field)
        if isinstance(tool_input.get("metadata"), dict):
            updates["metadata"] = tool_input["metadata"]
            updated_fields.append("metadata")
        if status is not None and status != existing.status:
            if status not in ("pending", "in_progress", "completed"):
                return ToolResult(content=f"TaskUpdate failed: invalid status {status!r}", is_error=True)
            updates["status"] = status
            updated_fields.append("status")

        try:
            task = store.update(existing.id, **updates) if updates else existing
            for block_id in list(tool_input.get("addBlocks") or []):
                if store.block_task(existing.id, str(block_id)):
                    if "blocks" not in updated_fields:
                        updated_fields.append("blocks")
            for blocker_id in list(tool_input.get("addBlockedBy") or []):
                if store.block_task(str(blocker_id), existing.id):
                    if "blockedBy" not in updated_fields:
                        updated_fields.append("blockedBy")
            task = store.require(existing.id)
            if status == "completed" and context.hook_engine and hasattr(context.hook_engine, "fire_task_completed"):
                hook_result = await context.hook_engine.fire_task_completed(task=task.to_dict(), session_id=context.session_id)
                if hook_result.should_block:
                    store.update(existing.id, status=existing.status)
                    return ToolResult(content=hook_result.stop_reason or "TaskUpdate completion blocked by hook", is_error=True)
            await self._emit_task_state(context, store)
        except Exception as exc:
            return ToolResult(content=f"TaskUpdate failed: {exc}", is_error=True)

        result = f"Updated task #{existing.id} {', '.join(updated_fields) if updated_fields else 'no fields changed'}"
        if status == "completed":
            result += "\n\nTask completed. Call TaskList now to find the next available task or see if your work unblocked others."
        return self._result(
            result,
            success=True,
            taskId=existing.id,
            updatedFields=updated_fields,
            statusChange={"from": existing.status, "to": status} if status and status != existing.status else None,
            task=task.to_dict() if isinstance(task, TaskRecord) else None,
            taskListId=store.task_list_id,
        )


class TaskClaimTool(_TaskToolBase):
    """Internal/runtime helper; not registered by default as a model-facing tool."""

    name = "TaskClaim"
    description = "Claim a task for an agent if it is unowned and unblocked."
    input_schema = {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "owner": {"type": "string"},
            "checkAgentBusy": {"type": "boolean"},
        },
        "required": ["taskId", "owner"],
    }

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        store = self._store_for_context(context)
        result = store.claim(
            str(tool_input.get("taskId") or ""),
            str(tool_input.get("owner") or context.session_id),
            check_agent_busy=bool(tool_input.get("checkAgentBusy")),
        )
        if not result.success:
            return self._result(
                f"TaskClaim failed: {result.reason}",
                success=False,
                reason=result.reason,
                busyWithTasks=result.busyWithTasks,
                blockedByTasks=result.blockedByTasks,
            )
        return self._result(f"Claimed task #{result.task.id}", success=True, task=result.task.to_dict() if result.task else None)


def task_system_tools(store: FileTaskStore | None = None) -> list[_TaskToolBase]:
    return [TaskCreateTool(store), TaskUpdateTool(store), TaskListTool(store), TaskGetTool(store)]
