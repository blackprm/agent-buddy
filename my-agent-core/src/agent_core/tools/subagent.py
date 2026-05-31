from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelClient
from agent_core.permissions.policy import PermissionPolicy
from agent_core.tools.base import ToolContext, ToolRegistry, ToolResult
from agent_core.types import TextBlock


class TaskTool:
    """Launch a synchronous subagent with fresh conversation context.

    This is the s06 Subagent migration: the parent receives only the child
    agent's final text summary; the child's intermediate messages are not
    appended to the parent conversation.  The child tool pool is supplied by
    the server factory and deliberately excludes Task to prevent recursion.
    """

    name = "Task"
    description = (
        "Launch a focused subagent to handle a complex, multi-step subtask with a fresh context. "
        "Use this for open-ended exploration, broad searches, independent analysis, or subtasks whose "
        "intermediate tool results would otherwise clutter the main conversation. The subagent shares the "
        "same working directory and filesystem side effects, but not the parent conversation history. "
        "It returns only its final concise conclusion. Do not use this for reading a single known file or "
        "for a task that can be handled by one direct tool call."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "A short 3-5 word description of the subtask"},
            "prompt": {"type": "string", "description": "The complete task briefing for the subagent"},
            "subagent_type": {
                "type": "string",
                "description": "Optional label for the subagent type. Currently informational; defaults to general-purpose.",
            },
        },
        "required": ["description", "prompt"],
    }
    is_concurrency_safe = False

    def __init__(
        self,
        *,
        model: ModelClient,
        sub_tools_factory: Callable[[], ToolRegistry],
        context_builder_factory: Callable[[str, str | None], ContextBuilder],
        permission_policy_factory: Callable[[], PermissionPolicy],
        max_turns: int = 30,
        max_depth: int = 1,
    ) -> None:
        self._model = model
        self._sub_tools_factory = sub_tools_factory
        self._context_builder_factory = context_builder_factory
        self._permission_policy_factory = permission_policy_factory
        self._max_turns = max_turns
        self._max_depth = max_depth

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        description = str(tool_input.get("description") or "subtask").strip()
        prompt = str(tool_input.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(
                content="Task failed: missing required input field 'prompt'. Provide a complete subagent briefing.",
                is_error=True,
            )

        depth = int((context.metadata or {}).get("subagent_depth", 0) or 0)
        if depth >= self._max_depth:
            return ToolResult(
                content="Task failed: recursive subagents are disabled in this runtime.",
                is_error=True,
            )

        child_session_id = f"{context.session_id}:subagent:{uuid.uuid4().hex[:8]}"
        subagent_type = str(tool_input.get("subagent_type") or "general-purpose")
        child_metadata = {
            **(context.metadata or {}),
            "agent_id": child_session_id,
            "parent_session_id": context.session_id,
            "subagent_depth": depth + 1,
            "subagent_type": subagent_type,
            "cwd": context.cwd,
        }

        hook_engine = context.hook_engine.clone() if hasattr(context.hook_engine, "clone") else context.hook_engine

        async def deny_child_permission(tool_name: str, tool_input: dict[str, Any]) -> str:
            return "deny"

        child_runtime = AgentRuntime(
            model=self._model,
            tools=self._sub_tools_factory(),
            context_builder=self._context_builder_factory(child_session_id, subagent_type),
            permission_policy=self._permission_policy_factory(),
            config=AgentRuntimeConfig(
                session_id=child_session_id,
                max_turns=self._max_turns,
                metadata=child_metadata,
                session_memory_enabled=False,
                cwd=context.cwd,
            ),
            session_store=None,
            # Synchronous Task currently returns only a final tool result to the
            # parent; it cannot stream child permission_request events to the UI.
            # Deny ask-only child tools instead of hanging on an invisible prompt.
            ask_callback=deny_child_permission,
            hook_engine=hook_engine,
        )

        final_text = ""
        terminal_status = "unknown"
        event_count = 0
        async for event in child_runtime.run(prompt):
            event_count += 1
            if event.type == "assistant_text":
                final_text = str(event.data.get("text") or final_text)
            elif event.type in ("loop_completed", "loop_failed", "loop_aborted"):
                terminal_status = event.type
                if event.type != "loop_completed" and not final_text:
                    final_text = str(event.data.get("error") or event.data.get("reason") or event.data)

        if not final_text:
            for msg in reversed(child_runtime.messages):
                if msg.role != "assistant":
                    continue
                text_parts = [block.text for block in msg.content if isinstance(block, TextBlock) and block.text]
                if text_parts:
                    final_text = "\n".join(text_parts)
                    break

        if not final_text:
            final_text = "Subagent finished without a final text response."

        is_error = terminal_status in ("loop_failed", "loop_aborted")
        return ToolResult(
            content=final_text,
            is_error=is_error,
            metadata={
                "status": "completed" if not is_error else terminal_status,
                "description": description,
                "subagentType": subagent_type,
                "childSessionId": child_session_id,
                "events": event_count,
            },
        )
