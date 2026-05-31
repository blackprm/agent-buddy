from __future__ import annotations

from typing import Any
import re
import uuid

from agent_core.skills.store import SkillStore
from agent_core.tools.base import ToolContext, ToolResult


class SkillTool:
    name = "Skill"
    description = (
        "Execute a local skill within the main conversation. Use this before responding "
        "when the user's request clearly matches an available skill."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Skill name to invoke"},
            "args": {"type": "string", "description": "Optional raw argument string"},
            "arguments": {
                "type": "array",
                "description": "Optional positional arguments",
                "items": {"type": "string"},
            },
        },
        "required": ["skill"],
    }
    is_concurrency_safe = True

    def __init__(self, store: SkillStore, *, model: Any | None = None, tools_factory: Any | None = None, context_builder_factory: Any | None = None, permission_policy_factory: Any | None = None) -> None:
        self._store = store
        self._model = model
        self._tools_factory = tools_factory
        self._context_builder_factory = context_builder_factory
        self._permission_policy_factory = permission_policy_factory

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        name = str(tool_input.get("skill") or tool_input.get("name") or "").strip()
        if not name:
            return ToolResult(content="Skill failed: missing required input field 'skill'", is_error=True)
        if name.startswith("/"):
            name = name[1:]
        args = tool_input.get("args")
        if args is None and isinstance(tool_input.get("arguments"), list):
            args = " ".join(str(a) for a in tool_input["arguments"])
        skill = self._store.get_skill(name)
        if skill is None:
            available = ", ".join(s.name for s in self._store.list_skills()) or "none"
            return ToolResult(content=f"Unknown skill: {name}. Available skills: {available}", is_error=True, metadata={"errorCode": 2})
        if skill.disable_model_invocation:
            return ToolResult(content=f"Skill {name} cannot be used with Skill tool due to disable-model-invocation", is_error=True, metadata={"skill": name, "errorCode": 4})
        try:
            content = self._store.render_skill(name, str(args or ""), session_id=context.session_id)
            content = await self._execute_prompt_shell(content, context, skill.allowed_tools, skill.shell)
        except FileNotFoundError as exc:
            return ToolResult(content=str(exc), is_error=True)
        except Exception as exc:
            return ToolResult(content=f"Skill {name} failed while rendering: {exc}", is_error=True)

        if skill.execution_context == "fork" and self._model and self._tools_factory and self._context_builder_factory:
            return await self._run_forked_skill(skill, content, context, str(args or ""))

        return ToolResult(
            content=content,
            metadata={
                "success": True,
                "skill": skill.name,
                "commandName": skill.name,
                "args": args or "",
                "loaded": True,
                "status": "inline",
                "allowedTools": skill.allowed_tools,
                "model": skill.model,
                "source": skill.source,
                "path": str(skill.path),
            },
        )

    async def _run_forked_skill(self, skill: Any, content: str, parent_context: ToolContext, args: str) -> ToolResult:
        from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
        from agent_core.types import TextBlock

        child_session_id = f"{parent_context.session_id}-skill-{skill.name}-{uuid.uuid4().hex[:8]}"
        builder = self._context_builder_factory(child_session_id, skill.agent or "general-purpose")
        policy = self._permission_policy_factory() if self._permission_policy_factory else None
        runtime = AgentRuntime(
            model=self._model,
            tools=self._tools_factory(),
            context_builder=builder,
            permission_policy=policy,
            config=AgentRuntimeConfig(session_id=child_session_id, max_turns=20, session_memory_enabled=False, skill_store=self._store),
            ask_callback=parent_context.ask_callback,
            hook_engine=parent_context.hook_engine,
            session_memory=None,
        )
        final_text = ""
        async for event in runtime.run(content):
            if event.type == "assistant_text":
                final_text = event.data.get("text", final_text)
        if not final_text:
            for msg in reversed(runtime.messages):
                if msg.role == "assistant":
                    parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                    final_text = "\n".join(parts).strip()
                    if final_text:
                        break
        return ToolResult(
            content=final_text or "Skill execution completed",
            metadata={
                "success": True,
                "skill": skill.name,
                "commandName": skill.name,
                "args": args,
                "status": "forked",
                "agent": skill.agent or "general-purpose",
                "childSessionId": child_session_id,
            },
        )

    async def _execute_prompt_shell(self, content: str, context: ToolContext, allowed_tools: list[str], shell: str | None) -> str:
        if "```!" not in content and "!`" not in content:
            return content
        if shell and shell not in {"bash", "sh"}:
            raise ValueError(f"Unsupported skill shell: {shell}")
        if allowed_tools and not _allowed_tools_include_bash(allowed_tools):
            raise PermissionError("Skill prompt shell execution requires allowed-tools to include bash")
        from agent_core.tools.builtin import BashTool
        from agent_core.permissions.policy import StaticPermissionPolicy

        bash_tool = BashTool()
        policy = self._permission_policy_factory() if self._permission_policy_factory else StaticPermissionPolicy(session_id=context.session_id)

        async def run_command(command: str) -> str:
            tool_input = {"command": command, "description": "skill prompt shell", "timeout": 120000}
            decision = await policy.check(tool=bash_tool, tool_input=tool_input)
            if decision.status == "deny":
                raise PermissionError(decision.reason or "Permission denied")
            if decision.status == "ask":
                if not context.ask_callback:
                    raise PermissionError(decision.reason or "Permission required for skill prompt shell command")
                try:
                    response = await context.ask_callback(bash_tool.name, tool_input, decision)
                except TypeError:
                    response = await context.ask_callback(bash_tool.name, tool_input)
                user_decision = response.get("decision") if isinstance(response, dict) else response
                user_option = response.get("option") if isinstance(response, dict) else None
                if user_decision != "allow":
                    raise PermissionError(decision.reason or "Permission denied")
                policy.record_user_decision(tool=bash_tool, tool_input=tool_input, option=user_option)
            result = await bash_tool.call(tool_input, context)
            if result.is_error:
                raise RuntimeError(result.content)
            stdout = str(result.metadata.get("stdout") or "").strip()
            stderr = str(result.metadata.get("stderr") or "").strip()
            if stdout and stderr:
                return f"{stdout}\n[stderr]\n{stderr}"
            return stdout or stderr or result.content

        block_pattern = re.compile(r"```!\s*\n?([\s\S]*?)\n?```", re.MULTILINE)
        inline_pattern = re.compile(r"(?:(?<=\s)|^)!`([^`]+)`", re.MULTILINE)
        for match in list(block_pattern.finditer(content)):
            command = (match.group(1) or "").strip()
            if command:
                output = await run_command(command)
                content = content.replace(match.group(0), output)
        for match in list(inline_pattern.finditer(content)):
            command = (match.group(1) or "").strip()
            if command:
                output = await run_command(command)
                content = content.replace(match.group(0), output)
        return content


def _allowed_tools_include_bash(allowed_tools: list[str]) -> bool:
    for raw in allowed_tools:
        token = str(raw).strip().lower()
        if token in {"bash", "bashtool"} or token.startswith("bash(") or token.startswith("bash:"):
            return True
    return False
