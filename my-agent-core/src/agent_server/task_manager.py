from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
from typing import Any

from agent_core.core.agent import AgentRuntime
from agent_core.core.events import AgentEvent


@dataclass
class SessionAgentTask:
    """A session-scoped background agent run.

    The task outlives any individual WebSocket connection.  WebSocket clients
    attach by replaying events from ``events`` after their last seen sequence,
    then wait on ``condition`` for new events.
    """

    session_id: str
    runtime: AgentRuntime
    message: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    events: Deque[tuple[int, AgentEvent | None]] = field(default_factory=lambda: deque(maxlen=2000))
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    seq: int = 0
    task: asyncio.Task | None = None
    permission_future: asyncio.Future[dict] | None = None
    pending_permission_request: dict[str, Any] | None = None
    pending_permission_response: dict | None = None
    workspace_manager: Any | None = None
    workspace: Any | None = None
    task_store: Any | None = None
    task_id: str | None = None
    done: bool = False
    terminal_sent: bool = False

    async def append_event(self, event: AgentEvent | None) -> None:
        async with self.condition:
            self.seq += 1
            if event and event.type == "permission_request":
                self.pending_permission_request = dict(event.data)
            elif event and event.type == "permission_result":
                self.pending_permission_request = None
            self.events.append((self.seq, event))
            self.condition.notify_all()


class AgentTaskManager:
    """Session-level background task manager for terminal WebSocket runs."""

    def __init__(self) -> None:
        self._tasks: dict[str, SessionAgentTask] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> SessionAgentTask | None:
        async with self._lock:
            return self._tasks.get(session_id)

    async def is_running(self, session_id: str) -> bool:
        task = await self.get(session_id)
        return bool(task and not task.done)

    async def start(
        self,
        *,
        session_id: str,
        runtime: AgentRuntime,
        message: str,
        attachments: list[dict[str, Any]] | None = None,
        workspace_manager: Any | None = None,
        workspace: Any | None = None,
        task_store: Any | None = None,
        task_id: str | None = None,
    ) -> SessionAgentTask | None:
        async with self._lock:
            existing = self._tasks.get(session_id)
            if existing and not existing.done:
                return None

            managed = SessionAgentTask(
                session_id=session_id,
                runtime=runtime,
                message=message,
                attachments=list(attachments or []),
                workspace_manager=workspace_manager,
                workspace=workspace,
                task_store=task_store,
                task_id=task_id,
            )

            async def ask_callback(tool_name: str, tool_input: dict, decision=None) -> dict:
                loop = asyncio.get_running_loop()
                if managed.pending_permission_response is not None:
                    response = managed.pending_permission_response
                    managed.pending_permission_response = None
                    await managed.append_event(AgentEvent("permission_result", {
                        "tool": tool_name,
                        "decision": response.get("decision", "deny"),
                        "option": response.get("option"),
                    }))
                    return response
                managed.permission_future = loop.create_future()
                try:
                    response = await managed.permission_future
                    await managed.append_event(AgentEvent("permission_result", {
                        "tool": tool_name,
                        "decision": response.get("decision", "deny"),
                        "option": response.get("option"),
                    }))
                    return response
                finally:
                    managed.permission_future = None

            runtime._ask_callback = ask_callback
            managed.task = asyncio.create_task(self._run(managed), name=f"agent-session-{session_id}")
            self._tasks[session_id] = managed
            return managed

    async def respond_permission(self, session_id: str, decision: str, option: dict | None = None) -> bool:
        task = await self.get(session_id)
        if not task or task.done:
            return False
        normalized = "allow" if decision == "allow" else "deny"
        response = {"decision": normalized, "option": option or ({"type": "accept-once"} if normalized == "allow" else {"type": "reject"})}
        if not task.permission_future or task.permission_future.done():
            if task.pending_permission_request is None:
                return False
            task.pending_permission_request = None
            task.pending_permission_response = response
            return True
        task.pending_permission_request = None
        task.permission_future.set_result(response)
        return True

    async def pending_permission(self, session_id: str) -> dict[str, Any] | None:
        task = await self.get(session_id)
        if not task or task.done or not task.pending_permission_request:
            return None
        return dict(task.pending_permission_request)

    async def abort(self, session_id: str) -> bool:
        task = await self.get(session_id)
        if not task or task.done:
            return False
        task.abort_event.set()
        record_cancel = getattr(task.runtime, "record_user_cancellation", None)
        if callable(record_cancel):
            with contextlib.suppress(Exception):
                record_cancel(reason="external_abort", phase="task_manager_abort")
        task.pending_permission_request = None
        task.pending_permission_response = None
        if task.permission_future and not task.permission_future.done():
            task.permission_future.set_result({"decision": "deny", "option": {"type": "reject"}})
        if task.task and not task.task.done():
            task.task.cancel()
        return True

    async def events_after(self, session_id: str, after_seq: int) -> tuple[list[tuple[int, AgentEvent | None]], int, bool]:
        task = await self.get(session_id)
        if not task:
            return [], after_seq, False
        async with task.condition:
            events = [(seq, event) for seq, event in task.events if seq > after_seq]
            return events, task.seq, task.done

    async def wait_for_next(self, session_id: str, after_seq: int) -> bool:
        task = await self.get(session_id)
        if not task:
            return False
        async with task.condition:
            await task.condition.wait_for(lambda: task.seq > after_seq or task.done)
            return True

    async def _run(self, managed: SessionAgentTask) -> None:
        try:
            if managed.workspace:
                await managed.append_event(AgentEvent("workspace_state", {
                    "status": "active",
                    "label": "isolated",
                    "path": getattr(managed.workspace, "worktree_path", ""),
                    "task_id": managed.task_id,
                }))
            run_kwargs: dict[str, Any] = {"abort_event": managed.abort_event}
            if managed.attachments:
                run_kwargs["attachments"] = managed.attachments
            async for event in managed.runtime.run(managed.message, **run_kwargs):
                if event.type in ("loop_completed", "loop_failed", "loop_aborted"):
                    managed.terminal_sent = True
                await managed.append_event(event)
        except asyncio.CancelledError:
            record_cancel = getattr(managed.runtime, "record_user_cancellation", None)
            if callable(record_cancel):
                with contextlib.suppress(Exception):
                    record_cancel(reason="cancelled", phase="task_manager_cancelled")
            await managed.append_event(AgentEvent("loop_aborted", {"reason": "cancelled"}))
            managed.terminal_sent = True
        except Exception as exc:  # noqa: BLE001 - server boundary converts to event
            await managed.append_event(AgentEvent("loop_failed", {"error": str(exc)}))
            managed.terminal_sent = True
        finally:
            if managed.workspace and managed.workspace_manager:
                removed = False
                changed = True
                with contextlib.suppress(Exception):
                    removed, changed = await managed.workspace_manager.cleanup_if_clean(managed.workspace)
                status = "preserved" if changed else ("removed" if removed else "cleanup_failed")
                payload = {
                    **managed.workspace.to_dict(),
                    "status": status,
                    "label": "isolated",
                    "has_changes": changed,
                    "removed": removed,
                }
                if managed.task_store and managed.task_id:
                    with contextlib.suppress(Exception):
                        managed.task_store.update(managed.task_id, metadata={"workspace": payload})
                await managed.append_event(AgentEvent("workspace_state", payload))
            if not managed.terminal_sent:
                await managed.append_event(AgentEvent("loop_completed", {
                    "turns": None,
                    "stop_reason": "agent_task_finished",
                }))
            managed.done = True
            await managed.append_event(None)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = [t for t in self._tasks.values() if t.task and not t.task.done()]
        for managed in tasks:
            managed.abort_event.set()
            if managed.task:
                managed.task.cancel()
        for managed in tasks:
            if managed.task:
                with contextlib.suppress(asyncio.CancelledError):
                    await managed.task


agent_task_manager = AgentTaskManager()
