from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.core.events import AgentEvent
from agent_core.model.base import ModelClient
from agent_core.permissions.policy import PermissionPolicy
from agent_core.session.store import SessionStore
from agent_core.teams.store import TeamMessage, TeamStore, VALID_MESSAGE_TYPES, sanitize_name
from agent_core.tools.base import ToolContext, ToolRegistry, ToolResult
from agent_core.types import TextBlock


def _context_identity(context: ToolContext) -> tuple[str, str, str]:
    metadata = context.metadata or {}
    team_name = sanitize_name(str(metadata.get("team_name") or metadata.get("teamName") or ""))
    agent_name = sanitize_name(str(metadata.get("agent_name") or metadata.get("agentName") or metadata.get("agent_id") or ""))
    if not agent_name:
        agent_name = "team-lead"
    return team_name, agent_name, str(metadata.get("parent_session_id") or "")


def _team_summary(team) -> str:
    lines = [f"Team `{team.name}` ({len(team.members)} member(s))"]
    if team.description:
        lines.append(f"Description: {team.description}")
    for member in team.members:
        lines.append(
            f"- {member.name} ({member.role}) status={member.status} session={member.child_session_id or 'unassigned'}"
        )
        if member.last_result:
            lines.append(f"  last_result: {member.last_result[:500]}")
    return "\n".join(lines)


def _messages_to_text(messages: list[TeamMessage]) -> str:
    if not messages:
        return "No inbox messages."
    lines: list[str] = []
    for idx, msg in enumerate(messages, 1):
        request = f" request_id={msg.request_id}" if msg.request_id else ""
        summary = f" summary={msg.summary}" if msg.summary else ""
        lines.append(f"{idx}. [{msg.type}] from={msg.sender}{request}{summary}\n{msg.content}")
    return "\n\n".join(lines)


async def _emit(context: ToolContext, event_type: str, data: dict[str, Any]) -> None:
    if context.event_callback:
        await context.event_callback(AgentEvent(event_type, data))


class TeamCreateTool:
    name = "TeamCreate"
    description = (
        "Create or open a persistent named agent team. Use this before spawning named teammates. "
        "A team owns a roster, team-scoped tasks, and inboxes for inter-agent messages."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string", "description": "Stable team name, e.g. frontend-migration"},
            "description": {"type": "string", "description": "Optional team purpose"},
        },
        "required": ["team_name"],
    }
    is_concurrency_safe = False
    should_defer = True

    def __init__(self, store: TeamStore) -> None:
        self._store = store

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = sanitize_name(str(tool_input.get("team_name") or ""))
        if not name:
            return ToolResult(content="TeamCreate failed: team_name is required.", is_error=True)
        metadata = context.metadata or {}
        team = self._store.create_team(
            name=name,
            lead_session_id=context.session_id,
            description=str(tool_input.get("description") or ""),
            user_id=str(metadata.get("user_id") or ""),
            org_id=str(metadata.get("org_id") or ""),
        )
        await _emit(context, "team_state", {"team": team.to_dict(), "action": "created"})
        return ToolResult(
            content=f"Team `{team.name}` is ready. Spawn named teammates with Agent using team_name=`{team.name}` and name=<teammate>.",
            metadata={"team": team.to_dict()},
        )


class TeamListTool:
    name = "TeamList"
    description = "List persistent teams and teammate statuses for the current session."
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string", "description": "Optional team name to inspect"},
        },
    }
    is_concurrency_safe = True
    should_defer = True

    def __init__(self, store: TeamStore) -> None:
        self._store = store

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = sanitize_name(str(tool_input.get("team_name") or ""))
        if name:
            team = self._store.get_team(name)
            if team is None:
                return ToolResult(content=f"Team not found: {name}", is_error=True)
            return ToolResult(content=_team_summary(team), metadata={"teams": [team.to_dict()]})
        user_id = str((context.metadata or {}).get("user_id") or "")
        teams = self._store.list_teams(lead_session_id=context.session_id, user_id=user_id or None)
        if not teams:
            return ToolResult(content="No persistent teams for this session.", metadata={"teams": []})
        return ToolResult(content="\n\n".join(_team_summary(team) for team in teams), metadata={"teams": [t.to_dict() for t in teams]})


class TeamDeleteTool:
    name = "TeamDelete"
    description = (
        "Delete a persistent team and its inboxes. By default this refuses while active teammates exist; "
        "ask them to shutdown first or pass force=true."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string"},
            "force": {"type": "boolean", "description": "Delete even if teammates are active"},
        },
        "required": ["team_name"],
    }
    is_concurrency_safe = False
    should_defer = True

    def __init__(self, store: TeamStore) -> None:
        self._store = store

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = sanitize_name(str(tool_input.get("team_name") or ""))
        if not name:
            return ToolResult(content="TeamDelete failed: team_name is required.", is_error=True)
        try:
            deleted = self._store.delete_team(name, force=bool(tool_input.get("force")))
        except RuntimeError as exc:
            return ToolResult(content=f"TeamDelete refused: {exc}", is_error=True)
        await _emit(context, "team_state", {"team_name": name, "action": "deleted", "deleted": deleted})
        return ToolResult(content=f"Deleted team `{name}`." if deleted else f"Team `{name}` did not exist.", metadata={"deleted": deleted})


class SendMessageTool:
    name = "SendMessage"
    description = (
        "Send a message to a teammate inbox in a persistent team. Use to coordinate named agents. "
        "Use to='*' to broadcast plain messages. Structured protocol types support shutdown and plan approval flows."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string", "description": "Team name; defaults to current team for teammates"},
            "to": {"type": "string", "description": "Recipient teammate name, team-lead, or * for broadcast"},
            "content": {"type": "string", "description": "Message body"},
            "summary": {"type": "string", "description": "Short summary for long messages"},
            "msg_type": {"type": "string", "enum": list(VALID_MESSAGE_TYPES), "description": "Message/protocol type"},
            "request_id": {"type": "string", "description": "Protocol correlation id; generated when omitted for protocol messages"},
            "extra": {"type": "object", "description": "Optional structured metadata"},
        },
        "required": ["to", "content"],
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(self, store: TeamStore) -> None:
        self._store = store

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        current_team, sender, _ = _context_identity(context)
        team_name = sanitize_name(str(tool_input.get("team_name") or current_team))
        if not team_name:
            return ToolResult(content="SendMessage failed: team_name is required outside a teammate context.", is_error=True)
        recipient = str(tool_input.get("to") or "").strip()
        if not recipient:
            return ToolResult(content="SendMessage failed: to is required.", is_error=True)
        msg_type = str(tool_input.get("msg_type") or "message")
        request_id = str(tool_input.get("request_id") or "").strip() or None
        if msg_type != "message" and request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:10]}"
        extra = tool_input.get("extra") if isinstance(tool_input.get("extra"), dict) else {}
        try:
            sent = self._store.send_message(
                team_name=team_name,
                sender=sender,
                recipient=recipient,
                content=str(tool_input.get("content") or ""),
                msg_type=msg_type,
                request_id=request_id,
                summary=str(tool_input.get("summary")) if tool_input.get("summary") else None,
                extra=extra,
            )
        except (KeyError, ValueError) as exc:
            return ToolResult(content=f"SendMessage failed: {exc}", is_error=True)
        if msg_type == "shutdown_response" and sender != "team-lead" and "approve" in extra:
            self._store.update_member(
                team_name=team_name,
                member_name=sender,
                status="shutdown" if bool(extra.get("approve")) else "idle",
                metadata={"last_shutdown_request_id": request_id},
            )
        return ToolResult(
            content=f"Sent {msg_type} to {', '.join(sent) or 'no recipients'}" + (f" (request_id={request_id})" if request_id else ""),
            metadata={"teamName": team_name, "from": sender, "to": sent, "msgType": msg_type, "requestId": request_id},
        )


class ReadInboxTool:
    name = "ReadInbox"
    description = "Read and optionally drain the current agent/team-lead inbox for a persistent team."
    input_schema = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string", "description": "Team name; defaults to current team for teammates"},
            "recipient": {"type": "string", "description": "Inbox owner; defaults to current agent or team-lead"},
            "drain": {"type": "boolean", "description": "Whether to remove messages after reading; default true"},
        },
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(self, store: TeamStore) -> None:
        self._store = store

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        current_team, agent_name, _ = _context_identity(context)
        team_name = sanitize_name(str(tool_input.get("team_name") or current_team))
        recipient = sanitize_name(str(tool_input.get("recipient") or agent_name))
        if not team_name:
            return ToolResult(content="ReadInbox failed: team_name is required outside a teammate context.", is_error=True)
        messages = self._store.read_inbox(team_name=team_name, recipient=recipient, drain=tool_input.get("drain") is not False)
        return ToolResult(
            content=_messages_to_text(messages),
            metadata={"teamName": team_name, "recipient": recipient, "messages": [m.to_dict() for m in messages]},
        )


class AgentTool:
    name = "Agent"
    description = (
        "Launch an agent. Without name/team_name it behaves like a one-shot subagent with fresh context and returns only "
        "the final summary. With both name and team_name it creates or resumes a persistent named teammate with a mailbox; "
        "use run_in_background=true for fire-and-forget team work. Teammates can communicate with SendMessage and ReadInbox."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short 3-5 word task description"},
            "prompt": {"type": "string", "description": "Complete task briefing"},
            "subagent_type": {"type": "string", "description": "Agent role/type label"},
            "name": {"type": "string", "description": "Persistent teammate name; omit for one-shot subagent"},
            "team_name": {"type": "string", "description": "Persistent team name when spawning a teammate"},
            "run_in_background": {"type": "boolean", "description": "Return immediately while the teammate works in the background"},
        },
        "required": ["description", "prompt"],
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(
        self,
        *,
        team_store: TeamStore,
        model: ModelClient,
        sub_tools_factory: Callable[[], ToolRegistry],
        context_builder_factory: Callable[[str, str | None], ContextBuilder],
        permission_policy_factory: Callable[[], PermissionPolicy],
        session_store: SessionStore | None = None,
        max_turns: int = 30,
        max_depth: int = 1,
    ) -> None:
        self._team_store = team_store
        self._model = model
        self._sub_tools_factory = sub_tools_factory
        self._context_builder_factory = context_builder_factory
        self._permission_policy_factory = permission_policy_factory
        self._session_store = session_store
        self._max_turns = max_turns
        self._max_depth = max_depth
        self._background_tasks: set[asyncio.Task] = set()

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        prompt = str(tool_input.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(content="Agent failed: missing required input field 'prompt'.", is_error=True)
        depth = int((context.metadata or {}).get("subagent_depth", 0) or 0)
        if depth >= self._max_depth:
            return ToolResult(content="Agent failed: recursive agent spawning is disabled in this runtime.", is_error=True)

        name = sanitize_name(str(tool_input.get("name") or ""))
        team_name = sanitize_name(str(tool_input.get("team_name") or ""))
        if bool(name) != bool(team_name):
            return ToolResult(content="Agent failed: persistent teammates require both name and team_name; omit both for one-shot subagent.", is_error=True)

        subagent_type = str(tool_input.get("subagent_type") or ("teammate" if name else "general-purpose"))
        description = str(tool_input.get("description") or "agent").strip()
        run_in_background = bool(tool_input.get("run_in_background")) and bool(name)

        if name:
            metadata = context.metadata or {}
            team = self._team_store.get_team(team_name)
            if team is None:
                team = self._team_store.create_team(
                    name=team_name,
                    lead_session_id=context.session_id,
                    user_id=str(metadata.get("user_id") or ""),
                    org_id=str(metadata.get("org_id") or ""),
                    description=f"Auto-created for teammate {name}",
                )
            existing = team.find_member(name)
            child_session_id = (existing.child_session_id if existing else "") or f"{context.session_id}:team:{team_name}:{name}"
            self._team_store.upsert_member(
                team_name=team_name,
                member_name=name,
                role=subagent_type,
                child_session_id=child_session_id,
                status="working",
                prompt=prompt,
                metadata={"description": description},
            )
            await _emit(context, "team_state", {"team_name": team_name, "member": name, "status": "working"})
            if run_in_background:
                task = asyncio.create_task(
                    self._run_child(context, prompt, child_session_id, subagent_type, team_name=team_name, agent_name=name),
                    name=f"agent-teammate-{team_name}-{name}",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                return ToolResult(
                    content=f"Started teammate `{name}` in team `{team_name}` in the background. Use TeamList and ReadInbox to monitor results.",
                    metadata={"teamName": team_name, "agentName": name, "childSessionId": child_session_id, "background": True},
                )
            final_text, status, event_count = await self._run_child(
                context,
                prompt,
                child_session_id,
                subagent_type,
                team_name=team_name,
                agent_name=name,
            )
            return ToolResult(
                content=final_text,
                is_error=status != "loop_completed",
                metadata={
                    "status": "completed" if status == "loop_completed" else status,
                    "description": description,
                    "subagentType": subagent_type,
                    "teamName": team_name,
                    "agentName": name,
                    "childSessionId": child_session_id,
                    "events": event_count,
                },
            )

        child_session_id = f"{context.session_id}:subagent:{uuid.uuid4().hex[:8]}"
        final_text, status, event_count = await self._run_child(context, prompt, child_session_id, subagent_type)
        return ToolResult(
            content=final_text,
            is_error=status != "loop_completed",
            metadata={
                "status": "completed" if status == "loop_completed" else status,
                "description": description,
                "subagentType": subagent_type,
                "childSessionId": child_session_id,
                "events": event_count,
            },
        )

    async def _run_child(
        self,
        parent_context: ToolContext,
        prompt: str,
        child_session_id: str,
        subagent_type: str,
        *,
        team_name: str = "",
        agent_name: str = "",
    ) -> tuple[str, str, int]:
        depth = int((parent_context.metadata or {}).get("subagent_depth", 0) or 0)
        child_metadata = {
            **(parent_context.metadata or {}),
            "agent_id": agent_name or child_session_id,
            "agent_name": agent_name,
            "team_name": team_name,
            "parent_session_id": parent_context.session_id,
            "subagent_depth": depth + 1,
            "subagent_type": subagent_type,
            "cwd": parent_context.cwd,
        }
        hook_engine = parent_context.hook_engine.clone() if hasattr(parent_context.hook_engine, "clone") else parent_context.hook_engine

        async def deny_child_permission(tool_name: str, tool_input: dict[str, Any], *_args: Any) -> str:
            return "deny"

        builder = self._context_builder_factory(child_session_id, subagent_type)
        if team_name and agent_name:
            builder.append_prompt = (builder.append_prompt + "\n\n" if builder.append_prompt else "") + (
                "# Persistent Teammate Instructions\n"
                f"You are `{agent_name}` in team `{team_name}`. Your role is `{subagent_type}`.\n"
                "Use SendMessage to communicate with team-lead or teammates. Use ReadInbox to check messages. "
                "When finished, report a concise final result; your status and result will be saved."
            )
        child_runtime = AgentRuntime(
            model=self._model,
            tools=self._sub_tools_factory(),
            context_builder=builder,
            permission_policy=self._permission_policy_factory(),
            config=AgentRuntimeConfig(
                session_id=child_session_id,
                max_turns=self._max_turns,
                metadata=child_metadata,
                session_memory_enabled=False,
                cwd=parent_context.cwd,
            ),
            session_store=self._session_store if team_name else None,
            ask_callback=deny_child_permission,
            hook_engine=hook_engine,
        )

        inbox_text = ""
        if team_name and agent_name:
            inbox = self._team_store.read_inbox(team_name=team_name, recipient=agent_name, drain=True)
            if inbox:
                inbox_text = "\n\n[Initial teammate inbox]\n" + _messages_to_text(inbox)
        final_text = ""
        terminal_status = "unknown"
        event_count = 0
        try:
            async for event in child_runtime.run(prompt + inbox_text):
                event_count += 1
                if event.type == "assistant_text":
                    final_text = str(event.data.get("text") or final_text)
                elif event.type in ("loop_completed", "loop_failed", "loop_aborted"):
                    terminal_status = event.type
                    if event.type != "loop_completed" and not final_text:
                        final_text = str(event.data.get("error") or event.data.get("reason") or event.data)
        except Exception as exc:  # noqa: BLE001 - background child boundary
            terminal_status = "loop_failed"
            final_text = repr(exc)

        if not final_text:
            for msg in reversed(child_runtime.messages):
                if msg.role != "assistant":
                    continue
                text_parts = [block.text for block in msg.content if isinstance(block, TextBlock) and block.text]
                if text_parts:
                    final_text = "\n".join(text_parts)
                    break
        if not final_text:
            final_text = "Agent finished without a final text response."

        if team_name and agent_name:
            self._team_store.update_member(
                team_name=team_name,
                member_name=agent_name,
                status="idle" if terminal_status == "loop_completed" else "failed",
                last_result=final_text,
                metadata={"terminal_status": terminal_status, "events": event_count},
            )
            self._team_store.send_message(
                team_name=team_name,
                sender=agent_name,
                recipient="team-lead",
                content=final_text,
                msg_type="message",
                summary=f"{agent_name} finished with {terminal_status}",
            )
        return final_text, terminal_status, event_count
