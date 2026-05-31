"""Coding Agent 配套工具集。"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.sandbox import get_sandbox_manager
from agent_core.tools.base import ToolContext, ToolRegistry, ToolResult


# ── TodoWrite 状态（源码中存在 AppState.todos[todoKey]；这里按 session_id 存内存）──

TodoItem = dict[str, Any]
_TODOS_BY_SESSION: dict[str, list[TodoItem]] = {}


TODO_WRITE_PROMPT = """Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. Task States:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

2. Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done, e.g. "Run tests"
   - activeForm: The present continuous form shown during execution, e.g. "Running tests"

3. Task Management:
   - Update task status in real time as you work
   - Mark tasks complete immediately after finishing
   - Ideally keep exactly one task in_progress at a time
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

4. Task Completion Requirements:
   - Only mark a task as completed when it is fully accomplished
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - Never mark a task as completed if tests are failing, implementation is partial, or errors are unresolved

When in doubt for a non-trivial task, use this tool. It does not perform work; it updates the planning state for the current session.
"""


# ── Echo（调试用）────────────────────────────────────────────


class EchoTool:
    name = "echo"
    description = "Echo back a text string. Useful for smoke tests."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    is_concurrency_safe = True
    should_defer = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        return ToolResult(content=str(tool_input.get("text", "")))


class ToolSearchTool:
    name = "ToolSearch"
    description = (
        "Search and activate deferred tools. Use this when you need a tool that is not currently visible, "
        "such as persistent Task System tools. Pass names to activate exact tools, or omit names to list all deferred tools."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms for deferred tools"},
            "names": {"type": "array", "items": {"type": "string"}, "description": "Exact tool names to activate"},
        },
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry

    def bind(self, registry: ToolRegistry) -> "ToolSearchTool":
        self._registry = registry
        return self

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        if self._registry is None:
            return ToolResult(content="ToolSearch failed: registry is not bound", is_error=True)
        query = str(tool_input.get("query") or "").lower().strip()
        names = [str(name) for name in (tool_input.get("names") or []) if str(name).strip()]
        deferred = self._registry.deferred_schemas()
        if query:
            deferred = [
                schema for schema in deferred
                if query in schema["name"].lower() or query in schema["description"].lower()
            ]
        activated = self._registry.activate_deferred(names or [schema["name"] for schema in deferred])
        lines = ["Deferred tools:"]
        for schema in deferred:
            status = "activated" if schema["name"] in activated or schema.get("activated") else "available"
            lines.append(f"- {schema['name']} ({status}): {schema['description'].splitlines()[0]}")
        if activated:
            lines.append("\nActivated tools will be visible on the next model turn: " + ", ".join(activated))
        return ToolResult(content="\n".join(lines), metadata={"deferred_tools": deferred, "activated": activated})


# ── TodoWrite（规划状态工具）──────────────────────────────────


class TodoWriteTool:
    name = "TodoWrite"
    description = (
        "Update the todo list for the current session. To be used proactively and often to track progress "
        "and pending tasks. Make sure that at least one task is in_progress when active work is ongoing. "
        "Always provide both content (imperative) and activeForm (present continuous) for each task.\n\n"
        + TODO_WRITE_PROMPT
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The updated todo list",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1, "description": "Imperative task description"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "activeForm": {"type": "string", "minLength": 1, "description": "Present continuous form shown during execution"},
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    }
    is_concurrency_safe = False
    should_defer = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        todos = tool_input.get("todos")
        if not isinstance(todos, list):
            return ToolResult(content="TodoWrite failed: 'todos' must be an array", is_error=True)

        normalized: list[TodoItem] = []
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                return ToolResult(content=f"TodoWrite failed: todos[{i}] must be an object", is_error=True)
            content = item.get("content")
            status = item.get("status")
            active_form = item.get("activeForm")
            if not isinstance(content, str) or not content.strip():
                return ToolResult(content=f"TodoWrite failed: todos[{i}].content cannot be empty", is_error=True)
            if status not in ("pending", "in_progress", "completed"):
                return ToolResult(content=f"TodoWrite failed: todos[{i}].status is invalid: {status!r}", is_error=True)
            if not isinstance(active_form, str) or not active_form.strip():
                return ToolResult(content=f"TodoWrite failed: todos[{i}].activeForm cannot be empty", is_error=True)
            normalized.append({
                "content": content.strip(),
                "status": status,
                "activeForm": active_form.strip(),
            })

        todo_key = context.metadata.get("agent_id") or context.session_id
        old_todos = list(_TODOS_BY_SESSION.get(todo_key, []))

        all_done = bool(normalized) and all(t["status"] == "completed" for t in normalized)
        if all_done:
            _TODOS_BY_SESSION[todo_key] = []
        else:
            _TODOS_BY_SESSION[todo_key] = normalized

        verification_nudge_needed = (
            all_done
            and len(normalized) >= 3
            and not any("verif" in t["content"].lower() or "验证" in t["content"] or "测试" in t["content"] for t in normalized)
        )

        content = "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress. Please proceed with the current tasks if applicable."
        if verification_nudge_needed:
            content += (
                "\n\nNOTE: You just closed out 3+ tasks and none of them was a verification step. "
                "Before writing your final summary, perform an explicit verification step such as running tests, builds, or a focused smoke check."
            )

        return ToolResult(
            content=content,
            metadata={
                "oldTodos": old_todos,
                "newTodos": normalized,
                "storedTodos": _TODOS_BY_SESSION[todo_key],
                "verificationNudgeNeeded": verification_nudge_needed,
            },
        )


def get_session_todos(session_id: str) -> list[TodoItem]:
    """返回当前 session 的 TodoWrite 内存状态，供 UI / 调试接口读取。"""
    return list(_TODOS_BY_SESSION.get(session_id, []))


# ── Plan Mode ────────────────────────────────────────────────


ENTER_PLAN_MODE_PROMPT = """Requests permission to enter Plan Mode for complex tasks requiring exploration and design.

Use this when the user asks you to plan first, design an approach before coding, or when the task is complex enough that implementation should not begin immediately.
"""


EXIT_PLAN_MODE_PROMPT = """Use this tool when you are in Plan Mode, have written the plan to the plan file, and are ready for user approval.

Do not use this tool for pure research. Use it only when the next step would be implementation after the user approves the plan.
"""


class EnterPlanModeTool:
    name = "EnterPlanMode"
    description = ENTER_PLAN_MODE_PROMPT
    input_schema = {"type": "object", "properties": {}}
    is_concurrency_safe = True
    should_defer = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        from agent_core.plan_mode import get_plan_file_path, prepare_context_for_plan_mode

        _, previous = prepare_context_for_plan_mode(context.session_id)
        plan_file = get_plan_file_path(context.session_id, cwd=context.cwd)
        message = (
            "Entered Plan Mode. Explore the codebase and design an implementation approach before coding.\n\n"
            f"Plan file: {plan_file}\n\n"
            "In Plan Mode:\n"
            "1. Read/search files and inspect the project thoroughly.\n"
            "2. Do not edit project files or run mutating commands.\n"
            "3. Write the final plan to the plan file.\n"
            "4. Call ExitPlanMode when the plan is ready for user approval."
        )
        return ToolResult(content=message, metadata={"planFilePath": str(plan_file), "previousMode": previous, "mode": "plan"})


class ExitPlanModeTool:
    name = "ExitPlanMode"
    description = EXIT_PLAN_MODE_PROMPT
    input_schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "string", "description": "Optional plan content. If omitted, the active plan file is read from disk."},
            "allowedPrompts": {
                "type": "array",
                "description": "Optional semantic permissions requested by the plan.",
                "items": {"type": "object"},
            },
        },
    }
    is_concurrency_safe = True
    should_defer = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        from agent_core.plan_mode import consume_plan_acceptance, exit_plan_mode, get_plan, get_plan_file_path, write_plan
        from agent_core.permissions.policy import get_session_permission_state

        state = get_session_permission_state(context.session_id)
        if state.mode == "plan":
            # Normal Web permission approval updates the mode before call().  This
            # fallback keeps non-interactive callers from getting trapped in plan mode.
            exit_plan_mode(context.session_id)
        plan_file = get_plan_file_path(context.session_id, cwd=context.cwd)
        input_plan = tool_input.get("plan") if isinstance(tool_input.get("plan"), str) else None
        if input_plan is not None:
            write_plan(context.session_id, input_plan, cwd=context.cwd)
        plan = input_plan if input_plan is not None else get_plan(context.session_id, cwd=context.cwd)
        if not plan or not plan.strip():
            return ToolResult(
                content=f"No plan found at {plan_file}. Write your plan to this file before calling ExitPlanMode.",
                is_error=True,
                metadata={"planFilePath": str(plan_file), "plan": plan},
            )

        acceptance = consume_plan_acceptance(context.session_id)
        mode = get_session_permission_state(context.session_id).mode
        clear_context = bool((acceptance or {}).get("clearContext", False))
        return ToolResult(
            content=(
                "Plan approved. Exit Plan Mode and proceed with implementation.\n\n"
                f"Permission mode: {mode}\n"
                f"Context handoff: {'clear context' if clear_context else 'keep context'}\n"
                f"Plan file: {plan_file}\n\n"
                f"Plan:\n{plan}"
            ),
            metadata={
                "plan": plan,
                "planFilePath": str(plan_file),
                "mode": mode,
                "allowedPrompts": tool_input.get("allowedPrompts"),
                "planImplementation": {
                    "plan": plan,
                    "planFilePath": str(plan_file),
                    "clearContext": clear_context,
                    "mode": mode,
                },
            },
        )


def _required_input(
    tool_input: dict[str, Any],
    field: str,
    *,
    aliases: tuple[str, ...] = (),
) -> tuple[Any | None, ToolResult | None]:
    """读取必填工具参数，缺失时返回对模型可恢复的明确错误。"""
    for key in (field, *aliases):
        value = tool_input.get(key)
        if value is not None and value != "":
            return value, None

    received = ", ".join(sorted(tool_input.keys())) or "none"
    alias_text = f" Accepted aliases: {', '.join(aliases)}." if aliases else ""
    return None, ToolResult(
        content=(
            f"missing required input field '{field}'. Received keys: {received}."
            f"{alias_text} Please retry this tool call with '{field}'."
        ),
        is_error=True,
        metadata={
            "missing_field": field,
            "received_keys": sorted(tool_input.keys()),
            "aliases": list(aliases),
        },
    )


# ── 读文件 ──────────────────────────────────────────────────


class ReadTextFileTool:
    name = "read_text_file"
    description = "Read a UTF-8 text file. Returns the file content."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read. Use this field for the target file path."},
            "file_path": {"type": "string", "description": "Alias for path; accepted for compatibility."},
            "offset": {"type": "integer", "description": "Line offset to start reading from (1-based). Default: 1"},
            "limit": {"type": "integer", "description": "Max number of lines to read. Default: read all"},
        },
        "required": ["path"],
    }
    is_concurrency_safe = True

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        try:
            raw_path, err = _required_input(tool_input, "path", aliases=("file_path", "filepath", "filename"))
            if err:
                return err
            path = context.resolve_path(str(raw_path))
            offset = int(tool_input.get("offset", 1))
            limit = tool_input.get("limit")
            content = path.read_text(encoding="utf-8")
            if offset > 1 or limit is not None:
                lines = content.splitlines()
                start = max(0, offset - 1)
                end = start + limit if limit else len(lines)
                content = "\n".join(lines[start:end])
                if offset > 1 or (limit and end < len(lines)):
                    content = f"(lines {start+1}-{min(end, len(lines))} of {len(lines)})\n{content}"
            return ToolResult(content=content)
        except Exception as exc:
            return ToolResult(content=f"read failed: {exc}", is_error=True)


# ── 写文件 ──────────────────────────────────────────────────


class WriteTextFileTool:
    name = "write_text_file"
    description = (
        "Write content to a UTF-8 text file. Creates the file and parent directories if they don't exist. "
        "Use this to create new files or completely overwrite existing ones."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "file_path": {"type": "string", "description": "Alias for path; accepted for compatibility."},
            "content": {"type": "string", "description": "File content to write"},
            "text": {"type": "string", "description": "Alias for content; accepted for compatibility."},
        },
        "required": ["path", "content"],
    }
    is_concurrency_safe = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        try:
            raw_path, err = _required_input(tool_input, "path", aliases=("file_path", "filepath", "filename"))
            if err:
                return err
            raw_content, err = _required_input(tool_input, "content", aliases=("text",))
            if err:
                return err
            path = context.resolve_path(str(raw_path))
            content = str(raw_content)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return ToolResult(content=f"Wrote {lines} lines to {path}")
        except Exception as exc:
            return ToolResult(content=f"write failed: {exc}", is_error=True)


# ── 编辑文件（搜索替换）──────────────────────────────────────


class EditFileTool:
    name = "edit_file"
    description = (
        "Edit a file by replacing exact text matches. Supports two modes: "
        "1) 'replace': replace old_string with new_string (fails if old_string not found or appears multiple times unless replace_all=true). "
        "2) 'insert': insert new_string at the specified line number (1-based). "
        "Prefer this over write_text_file for small targeted edits."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "file_path": {"type": "string", "description": "Alias for path; accepted for compatibility."},
            "old_string": {"type": "string", "description": "Text to find and replace (required for replace mode)"},
            "old_text": {"type": "string", "description": "Alias for old_string; accepted for compatibility."},
            "new_string": {"type": "string", "description": "Replacement text (required for replace mode)"},
            "new_text": {"type": "string", "description": "Alias for new_string; accepted for compatibility."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences. Default: false"},
            "mode": {"type": "string", "enum": ["replace", "insert"], "description": "Edit mode. Default: replace"},
            "insert_line": {"type": "integer", "description": "Line number to insert at (1-based, for insert mode)"},
            "new_string_insert": {"type": "string", "description": "Text to insert (for insert mode)"},
        },
        "required": ["path"],
    }
    is_concurrency_safe = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        try:
            raw_path, err = _required_input(tool_input, "path", aliases=("file_path", "filepath", "filename"))
            if err:
                return err
            path = context.resolve_path(str(raw_path))
            mode = tool_input.get("mode", "replace")

            if not path.exists():
                return ToolResult(content=f"File not found: {path}", is_error=True)

            content = path.read_text(encoding="utf-8")

            if mode == "insert":
                insert_line = tool_input.get("insert_line")
                new_text = tool_input.get("new_string_insert", tool_input.get("new_string", ""))
                if insert_line is None:
                    return ToolResult(content="insert_line is required for insert mode", is_error=True)

                lines = content.split("\n")
                idx = max(0, min(int(insert_line) - 1, len(lines)))
                lines.insert(idx, new_text)
                path.write_text("\n".join(lines), encoding="utf-8")
                return ToolResult(content=f"Inserted at line {insert_line} in {path}")

            else:  # replace
                old_string = tool_input.get("old_string", tool_input.get("old_text", ""))
                new_string = tool_input.get("new_string", tool_input.get("new_text", ""))
                if not old_string:
                    return ToolResult(
                        content=(
                            "missing required input field 'old_string' for replace mode. "
                            f"Received keys: {', '.join(sorted(tool_input.keys())) or 'none'}. "
                            "Accepted aliases: old_text."
                        ),
                        is_error=True,
                    )

                count = content.count(old_string)
                if count == 0:
                    return ToolResult(content=f"old_string not found in {path}", is_error=True)
                if count > 1 and not tool_input.get("replace_all", False):
                    return ToolResult(
                        content=f"old_string found {count} times in {path}. Use replace_all=true or provide more context.",
                        is_error=True,
                    )

                new_content = content.replace(old_string, new_string) if tool_input.get("replace_all") else content.replace(old_string, new_string, 1)
                path.write_text(new_content, encoding="utf-8")
                return ToolResult(content=f"Replaced {count} occurrence(s) in {path}")

        except Exception as exc:
            return ToolResult(content=f"edit failed: {exc}", is_error=True)


# ── 列出目录 ────────────────────────────────────────────────


class ListDirectoryTool:
    name = "list_directory"
    description = (
        "List files and directories at the given path. Returns names, types (file/dir), and sizes. "
        "Use this to explore project structure."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path. Default: current working directory"},
            "recursive": {"type": "boolean", "description": "List recursively. Default: false"},
            "max_depth": {"type": "integer", "description": "Max recursion depth for recursive mode. Default: 3"},
        },
    }
    is_concurrency_safe = True

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        raw_path = tool_input.get("path", tool_input.get("directory", "."))
        path = context.resolve_path(raw_path)
        recursive = tool_input.get("recursive", False)
        max_depth = tool_input.get("max_depth", 3)

        if not path.exists():
            return ToolResult(content=f"Path not found: {path}", is_error=True)
        if not path.is_dir():
            return ToolResult(content=f"Not a directory: {path}", is_error=True)

        lines: list[str] = []
        try:
            if recursive:
                for root, dirs, files in os.walk(path):
                    rel = Path(root).relative_to(path)
                    depth = len(rel.parts)
                    if depth >= max_depth:
                        dirs.clear()
                        continue
                    prefix = "  " * depth
                    for d in sorted(dirs):
                        # Skip common hidden/ignored dirs
                        if d.startswith(".") or d in ("__pycache__", "node_modules", ".git", ".venv", "venv"):
                            continue
                        lines.append(f"{prefix}{d}/")
                    for f in sorted(files):
                        lines.append(f"{prefix}{f}")
            else:
                for entry in sorted(path.iterdir()):
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir():
                        lines.append(f"{entry.name}/")
                    else:
                        size = entry.stat().st_size
                        lines.append(f"{entry.name}  ({_fmt_size(size)})")

            result = "\n".join(lines) if lines else "(empty directory)"
            return ToolResult(content=result)
        except Exception as exc:
            return ToolResult(content=f"list failed: {exc}", is_error=True)


# ── 搜索文件内容 (grep) ─────────────────────────────────────


class GrepTool:
    name = "grep"
    description = (
        "Search file contents with a regex pattern. Returns matching lines with file paths and line numbers. "
        "Supports include/exclude glob patterns for file filtering."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory or file to search in. Default: current working directory"},
            "include": {"type": "string", "description": "Glob pattern for files to include (e.g. '*.py'). Default: all files"},
            "exclude": {"type": "string", "description": "Glob pattern for files to exclude. Default: none"},
            "case_insensitive": {"type": "boolean", "description": "Case insensitive search. Default: false"},
            "max_results": {"type": "integer", "description": "Max number of results. Default: 50"},
        },
        "required": ["pattern"],
    }
    is_concurrency_safe = True

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        try:
            raw_pattern, err = _required_input(tool_input, "pattern", aliases=("query", "regex"))
            if err:
                return err
            pattern = str(raw_pattern)
            raw_path = tool_input.get("path", tool_input.get("directory", "."))
            path = context.resolve_path(raw_path)
            include = tool_input.get("include", "")
            exclude = tool_input.get("exclude", "")
            case_insensitive = tool_input.get("case_insensitive", False)
            max_results = tool_input.get("max_results", 50)
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(content=f"Invalid regex: {e}", is_error=True)

        results: list[str] = []
        try:
            files = _collect_files(path, include, exclude)
            for file_path in files:
                if len(results) >= max_results:
                    results.append(f"... (truncated at {max_results} results)")
                    break
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if len(results) >= max_results:
                            break
                        if regex.search(line):
                            rel = file_path.relative_to(path) if file_path.is_relative_to(path) else file_path
                            results.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                except Exception:
                    continue

            if not results:
                return ToolResult(content="No matches found")
            return ToolResult(content="\n".join(results))
        except Exception as exc:
            return ToolResult(content=f"grep failed: {exc}", is_error=True)


# ── 查找文件 (glob) ─────────────────────────────────────────


class GlobTool:
    name = "glob"
    description = (
        "Find files matching a glob pattern. Returns list of matching file paths. "
        "Example patterns: '**/*.py', 'src/**/*.ts', '*.json'"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path": {"type": "string", "description": "Base directory. Default: current working directory"},
            "max_results": {"type": "integer", "description": "Max number of results. Default: 100"},
        },
        "required": ["pattern"],
    }
    is_concurrency_safe = True

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        try:
            raw_pattern, err = _required_input(tool_input, "pattern", aliases=("glob", "query"))
            if err:
                return err
            pattern = str(raw_pattern)
            raw_path = tool_input.get("path", tool_input.get("directory", "."))
            path = context.resolve_path(raw_path)
            max_results = tool_input.get("max_results", 100)
            matches = sorted(path.glob(pattern))
            results: list[str] = []
            for m in matches[:max_results]:
                rel = m.relative_to(path) if m.is_relative_to(path) else m
                results.append(str(rel))
            if len(matches) > max_results:
                results.append(f"... ({len(matches) - max_results} more)")
            if not results:
                return ToolResult(content="No files matched the pattern")
            return ToolResult(content="\n".join(results))
        except Exception as exc:
            return ToolResult(content=f"glob failed: {exc}", is_error=True)


# ── 执行 Shell 命令 ─────────────────────────────────────────


DEFAULT_BASH_TIMEOUT_MS = 120_000
MAX_BASH_TIMEOUT_MS = 600_000
BASH_PROGRESS_INTERVAL_SECONDS = 1.0
BASH_OUTPUT_PREVIEW_CHARS = 30_000
BASH_OUTPUT_FILE_THRESHOLD_CHARS = 60_000


@dataclass(slots=True)
class _ShellSnapshot:
    stdout: str = ""
    stderr: str = ""
    total_bytes: int = 0
    total_lines: int = 0
    truncated_stdout: bool = False
    truncated_stderr: bool = False


@dataclass(slots=True)
class _BackgroundShellTask:
    task_id: str
    command: str
    description: str
    process: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    total_bytes: int = 0
    total_lines: int = 0
    returncode: int | None = None
    output_file_path: str | None = None
    capture_task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.returncode is None and self.process.returncode is None

    def snapshot(self) -> _ShellSnapshot:
        stdout = "".join(self.stdout)
        stderr = "".join(self.stderr)
        return _ShellSnapshot(stdout=stdout, stderr=stderr, total_bytes=self.total_bytes, total_lines=self.total_lines)


_BACKGROUND_SHELL_TASKS: dict[str, _BackgroundShellTask] = {}


def _normalize_timeout_ms(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_BASH_TIMEOUT_MS
    try:
        timeout = int(float(value))
    except (TypeError, ValueError):
        return DEFAULT_BASH_TIMEOUT_MS
    return max(1, min(timeout, MAX_BASH_TIMEOUT_MS))


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _tail_preview(text: str, limit: int = BASH_OUTPUT_PREVIEW_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def _write_large_output(command: str, stdout: str, stderr: str) -> tuple[str | None, int]:
    combined = stdout + stderr
    size = len(combined.encode("utf-8", errors="replace"))
    if len(combined) < BASH_OUTPUT_FILE_THRESHOLD_CHARS:
        return None, size
    fd, path = tempfile.mkstemp(prefix="my-agent-bash-", suffix=".log")
    with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"$ {command}\n\n")
        if stdout:
            f.write("--- stdout ---\n")
            f.write(stdout)
            if not stdout.endswith("\n"):
                f.write("\n")
        if stderr:
            f.write("--- stderr ---\n")
            f.write(stderr)
            if not stderr.endswith("\n"):
                f.write("\n")
    return path, size


def _interpret_command_result(command: str, exit_code: int, stdout: str, stderr: str) -> dict[str, Any]:
    base = _extract_semantic_command(command)
    if base in {"grep", "rg"}:
        return {"is_error": exit_code >= 2, "message": "No matches found" if exit_code == 1 else None}
    if base == "find":
        return {"is_error": exit_code >= 2, "message": "Some directories were inaccessible" if exit_code == 1 else None}
    if base == "diff":
        return {"is_error": exit_code >= 2, "message": "Files differ" if exit_code == 1 else None}
    if base in {"test", "["}:
        return {"is_error": exit_code >= 2, "message": "Condition is false" if exit_code == 1 else None}
    return {"is_error": exit_code != 0, "message": f"Command failed with exit code {exit_code}" if exit_code != 0 else None}


def _extract_semantic_command(command: str) -> str:
    # Heuristic only; permissions/security decisions live in policy.py.
    tail = re.split(r"\s*(?:\|\||&&|;|\|)\s*", command.strip())[-1]
    return tail.split()[0] if tail.split() else ""


async def _emit_tool_progress(context: ToolContext, data: dict[str, Any]) -> None:
    if not context.event_callback:
        return
    from agent_core.core.events import AgentEvent
    await context.event_callback(AgentEvent("tool_progress", {
        "tool_use_id": context.metadata.get("tool_use_id"),
        "tool": context.metadata.get("tool_name") or "bash",
        **data,
    }))


async def _read_stream_to_chunks(
    stream: asyncio.StreamReader | None,
    chunks: list[str],
    stream_name: str,
    progress: dict[str, Any],
    context: ToolContext | None = None,
    started_at: float | None = None,
) -> None:
    if stream is None:
        return
    last_emit = 0.0
    while True:
        data = await stream.read(4096)
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        chunks.append(text)
        progress["total_bytes"] = int(progress.get("total_bytes", 0)) + len(data)
        progress["total_lines"] = int(progress.get("total_lines", 0)) + text.count("\n")
        if context is not None and started_at is not None:
            now = time.time()
            if now - last_emit >= BASH_PROGRESS_INTERVAL_SECONDS:
                last_emit = now
                stdout = "".join(progress.get("stdout", []))
                stderr = "".join(progress.get("stderr", []))
                preview_source = stdout or stderr
                preview, truncated = _tail_preview(preview_source, 4_000)
                await _emit_tool_progress(context, {
                    "kind": "bash_output",
                    "stream": stream_name,
                    "output": preview,
                    "stdout": stdout[-4_000:],
                    "stderr": stderr[-4_000:],
                    "elapsedTimeSeconds": int(now - started_at),
                    "totalLines": progress.get("total_lines", 0),
                    "totalBytes": progress.get("total_bytes", 0),
                    "truncated": truncated,
                })


async def _capture_background_task(task: _BackgroundShellTask) -> None:
    progress = {
        "stdout": task.stdout,
        "stderr": task.stderr,
        "total_bytes": len(("".join(task.stdout) + "".join(task.stderr)).encode("utf-8", errors="replace")),
        "total_lines": ("".join(task.stdout) + "".join(task.stderr)).count("\n"),
    }
    readers = [
        asyncio.create_task(_read_stream_to_chunks(task.process.stdout, task.stdout, "stdout", progress)),
        asyncio.create_task(_read_stream_to_chunks(task.process.stderr, task.stderr, "stderr", progress)),
    ]
    try:
        await task.process.wait()
        await asyncio.gather(*readers, return_exceptions=True)
    finally:
        task.returncode = task.process.returncode
        task.ended_at = time.time()
        task.total_bytes = int(progress.get("total_bytes", 0))
        task.total_lines = int(progress.get("total_lines", 0))
        task.output_file_path, _ = _write_large_output(task.command, "".join(task.stdout), "".join(task.stderr))


def _register_background_task(
    command: str,
    description: str,
    process: asyncio.subprocess.Process,
    *,
    initial_stdout: list[str] | None = None,
    initial_stderr: list[str] | None = None,
) -> _BackgroundShellTask:
    task_id = f"bash-{uuid.uuid4().hex[:10]}"
    task = _BackgroundShellTask(task_id=task_id, command=command, description=description or command, process=process)
    if initial_stdout:
        task.stdout.extend(initial_stdout)
    if initial_stderr:
        task.stderr.extend(initial_stderr)
    task.capture_task = asyncio.create_task(_capture_background_task(task), name=f"background-{task_id}")
    _BACKGROUND_SHELL_TASKS[task_id] = task
    return task


class BashTool:
    name = "bash"
    description = (
        "Executes a given bash command in a persistent working directory context and returns stdout/stderr. "
        "Supports Claude-compatible inputs: command, description, timeout in milliseconds, and run_in_background. "
        "Long-running foreground commands emit progress events and are moved to the background on timeout. "
        "Do not use curl, wget, httpie, or similar shell commands for public web search or page fetching; use WebSearch or WebFetch instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "description": {"type": "string", "description": "Clear, concise description of what this command does"},
            "timeout": {"type": "number", "description": f"Timeout in milliseconds. Default: {DEFAULT_BASH_TIMEOUT_MS}, max: {MAX_BASH_TIMEOUT_MS}"},
            "run_in_background": {"type": "boolean", "description": "Run command in the background and return immediately"},
            "dangerously_disable_sandbox": {"type": "boolean", "description": "Dangerously run this command outside the sandbox when unsandboxed fallback is allowed"},
            "dangerouslyDisableSandbox": {"type": "boolean", "description": "Alias for dangerously_disable_sandbox"},
            "cwd": {"type": "string", "description": "Working directory. Default: current working directory"},
        },
        "required": ["command"],
    }
    is_concurrency_safe = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        process: asyncio.subprocess.Process | None = None
        try:
            raw_command, err = _required_input(tool_input, "command", aliases=("cmd", "shell_command"))
            if err:
                return err
            command = str(raw_command)
            description = str(tool_input.get("description") or command)
            timeout_ms = _normalize_timeout_ms(tool_input.get("timeout"))
            timeout_seconds = timeout_ms / 1000
            run_in_background = bool(tool_input.get("run_in_background", False))
            cwd = tool_input.get("cwd", tool_input.get("path"))
            run_cwd = str(context.resolve_path(cwd)) if cwd else (context.cwd or os.getcwd())
            sandbox = get_sandbox_manager()
            sandbox_input = {**tool_input, "command": command}
            sandboxed = sandbox.should_use_sandbox(sandbox_input)
            executed_command = sandbox.wrap_command(command, cwd=run_cwd) if sandboxed else command

            process = await asyncio.create_subprocess_shell(
                executed_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=run_cwd,
                executable="/bin/bash" if Path("/bin/bash").exists() else None,
                start_new_session=True,
            )

            if run_in_background:
                task = _register_background_task(command, description, process)
                return ToolResult(
                    content=f"Running in the background (task_id: {task.task_id})",
                    metadata={"backgroundTaskId": task.task_id, "command": command, "description": description, "sandboxed": sandboxed},
                )

            started_at = time.time()
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            progress = {"stdout": stdout_chunks, "stderr": stderr_chunks, "total_bytes": 0, "total_lines": 0}
            readers = [
                asyncio.create_task(_read_stream_to_chunks(process.stdout, stdout_chunks, "stdout", progress, context, started_at)),
                asyncio.create_task(_read_stream_to_chunks(process.stderr, stderr_chunks, "stderr", progress, context, started_at)),
            ]

            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                for reader in readers:
                    reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                task = _register_background_task(
                    command,
                    description,
                    process,
                    initial_stdout=stdout_chunks,
                    initial_stderr=stderr_chunks,
                )
                await _emit_tool_progress(context, {
                    "kind": "bash_backgrounded",
                    "backgroundTaskId": task.task_id,
                    "elapsedTimeSeconds": int(time.time() - started_at),
                    "timeoutMs": timeout_ms,
                })
                return ToolResult(
                    content=f"Command timed out after {_format_duration(timeout_seconds)} and is still running in the background (task_id: {task.task_id})",
                    metadata={
                        "backgroundTaskId": task.task_id,
                        "timeoutMs": timeout_ms,
                        "timedOut": True,
                        "command": command,
                        "description": description,
                    },
                )

            await asyncio.gather(*readers, return_exceptions=True)
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            code = int(process.returncode or 0)
            interpretation = _interpret_command_result(command, code, stdout, stderr)
            output_file_path, output_file_size = _write_large_output(command, stdout, stderr)
            if sandboxed:
                sandbox.cleanup_after_command()
            stdout_preview, stdout_truncated = _tail_preview(stdout)
            stderr_preview, stderr_truncated = _tail_preview(stderr)

            parts: list[str] = []
            if stdout_preview.strip():
                parts.append(stdout_preview.strip())
            if stderr_preview.strip():
                parts.append(f"STDERR:\n{stderr_preview.strip()}")
            if interpretation.get("message"):
                parts.append(str(interpretation["message"]))
            if code != 0:
                parts.append(f"Exit code: {code}")
            if output_file_path:
                parts.append(f"Full output written to: {output_file_path}")
            elif stdout_truncated or stderr_truncated:
                parts.append("Output was truncated to the most recent content.")

            output = "\n".join(parts) or "(no output)"
            return ToolResult(
                content=output,
                is_error=bool(interpretation.get("is_error")),
                metadata={
                    "command": command,
                    "description": description,
                    "sandboxed": sandboxed,
                    "dangerouslyDisableSandbox": bool(tool_input.get("dangerously_disable_sandbox") or tool_input.get("dangerouslyDisableSandbox")),
                    "stdout": stdout_preview,
                    "stderr": stderr_preview,
                    "exitCode": code,
                    "returnCodeInterpretation": interpretation.get("message"),
                    "elapsedMs": int((time.time() - started_at) * 1000),
                    "totalBytes": progress.get("total_bytes", 0),
                    "totalLines": progress.get("total_lines", 0),
                    "stdoutTruncated": stdout_truncated,
                    "stderrTruncated": stderr_truncated,
                    "outputFilePath": output_file_path,
                    "outputFileSize": output_file_size if output_file_path else None,
                },
            )
        except asyncio.TimeoutError:
            return ToolResult(content="Command timed out", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"bash failed: {exc}", is_error=True)


class BashOutputTool:
    name = "bash_output"
    description = "Retrieve stdout/stderr from a running or completed background bash task by task_id."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Background bash task id"},
            "bash_id": {"type": "string", "description": "Alias for task_id"},
        },
        "required": ["task_id"],
    }
    is_concurrency_safe = True

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        raw_task_id, err = _required_input(tool_input, "task_id", aliases=("bash_id", "shell_id"))
        if err:
            return err
        task_id = str(raw_task_id)
        task = _BACKGROUND_SHELL_TASKS.get(task_id)
        if not task:
            return ToolResult(content=f"Unknown background task: {task_id}", is_error=True)
        snapshot = task.snapshot()
        stdout_preview, stdout_truncated = _tail_preview(snapshot.stdout)
        stderr_preview, stderr_truncated = _tail_preview(snapshot.stderr)
        status = "running" if task.running else f"completed (exit {task.returncode})"
        parts = [f"Task {task_id}: {status}"]
        if stdout_preview.strip():
            parts.append(stdout_preview.strip())
        if stderr_preview.strip():
            parts.append(f"STDERR:\n{stderr_preview.strip()}")
        if task.output_file_path:
            parts.append(f"Full output written to: {task.output_file_path}")
        return ToolResult(
            content="\n".join(parts),
            is_error=(task.returncode is not None and task.returncode != 0),
            metadata={
                "backgroundTaskId": task_id,
                "status": status,
                "stdout": stdout_preview,
                "stderr": stderr_preview,
                "exitCode": task.returncode,
                "running": task.running,
                "stdoutTruncated": stdout_truncated,
                "stderrTruncated": stderr_truncated,
                "outputFilePath": task.output_file_path,
            },
        )


class KillShellTool:
    name = "kill_shell"
    description = "Terminate a running background bash task by task_id."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Background bash task id"},
            "shell_id": {"type": "string", "description": "Alias for task_id"},
        },
        "required": ["task_id"],
    }
    is_concurrency_safe = False

    async def call(self, tool_input: dict, context: ToolContext) -> ToolResult:
        raw_task_id, err = _required_input(tool_input, "task_id", aliases=("bash_id", "shell_id"))
        if err:
            return err
        task_id = str(raw_task_id)
        task = _BACKGROUND_SHELL_TASKS.get(task_id)
        if not task:
            return ToolResult(content=f"Unknown background task: {task_id}", is_error=True)
        if not task.running:
            return ToolResult(content=f"Task {task_id} is already completed (exit {task.returncode})")
        try:
            if task.process.pid:
                os.killpg(task.process.pid, signal.SIGTERM)
            else:
                task.process.terminate()
            return ToolResult(content=f"Terminated background task {task_id}", metadata={"backgroundTaskId": task_id})
        except ProcessLookupError:
            return ToolResult(content=f"Task {task_id} is no longer running")


# ── 辅助函数 ────────────────────────────────────────────────


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"


def _collect_files(
    path: Path,
    include: str = "",
    exclude: str = "",
) -> list[Path]:
    """收集要搜索的文件列表。"""
    _SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}

    if path.is_file():
        return [path]

    files: list[Path] = []
    for root, dirs, filenames in os.walk(path):
        # 跳过隐藏和忽略目录
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = Path(root) / fname
            if include and not fnmatch.fnmatch(fname, include):
                continue
            if exclude and fnmatch.fnmatch(fname, exclude):
                continue
            files.append(fpath)
    return files
