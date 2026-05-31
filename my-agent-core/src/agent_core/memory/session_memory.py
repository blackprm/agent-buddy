"""Session Memory — durable per-session Markdown notes.

This is a pragmatic Python migration of Claude Code's s09 memory behavior:

- keep one ``summary.md`` file per session under ``~/.my-agent-core/projects``;
- inject existing non-empty memory into the next system prompt;
- after useful turns, update the memory in an isolated background task;
- persist extraction thresholds/state in a sidecar ``state.json`` so reconnects and
  fresh Runtime objects keep the same cadence.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.model.base import ModelClient
from agent_core.types import Message, TextBlock, ToolUseBlock


DEFAULT_SESSION_MEMORY_TEMPLATE = """# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""


_REQUIRED_HEADERS = [
    "# Session Title",
    "# Current State",
    "# Task specification",
    "# Files and Functions",
    "# Workflow",
    "# Errors & Corrections",
    "# Codebase and System Documentation",
    "# Learnings",
    "# Key results",
    "# Worklog",
]


@dataclass(slots=True)
class SessionMemoryConfig:
    enabled: bool = True
    minimum_message_tokens_to_init: int = 10_000
    minimum_tokens_between_update: int = 5_000
    tool_calls_between_updates: int = 3
    extraction_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "SessionMemoryConfig":
        return cls(
            enabled=_env_bool("AGENT_SESSION_MEMORY_ENABLED", True),
            minimum_message_tokens_to_init=_env_int("AGENT_SESSION_MEMORY_MIN_INIT_TOKENS", 10_000),
            minimum_tokens_between_update=_env_int("AGENT_SESSION_MEMORY_MIN_UPDATE_TOKENS", 5_000),
            tool_calls_between_updates=_env_int("AGENT_SESSION_MEMORY_TOOL_CALLS", 3),
            extraction_timeout_seconds=float(os.getenv("AGENT_SESSION_MEMORY_TIMEOUT_SECONDS", "60")),
        )


class SessionMemoryManager:
    """Manage one session's durable memory file and extraction cadence."""

    def __init__(
        self,
        session_id: str,
        *,
        cwd: str | Path | None = None,
        base_dir: str | Path | None = None,
        config: SessionMemoryConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.config = config or SessionMemoryConfig.from_env()
        root = Path(base_dir or os.getenv("AGENT_SESSION_MEMORY_DIR") or (Path.home() / ".my-agent-core" / "projects"))
        self.memory_dir = root.expanduser() / _project_slug(self.cwd) / _session_slug(session_id) / "session-memory"
        self.memory_path = self.memory_dir / "summary.md"
        self.state_path = self.memory_dir / "state.json"
        self._lock = _lock_for(str(self.memory_path))

    async def context_section(self) -> str | None:
        """Return the dynamic system prompt section for existing non-empty memory."""
        if not self.config.enabled:
            return None
        content = await asyncio.to_thread(self.get_memory_content)
        if not content or self.is_template_only(content):
            return None
        return (
            "# Session Memory\n"
            "The following durable notes were extracted from earlier turns in this same session. "
            "Use them for continuity, but treat newer user messages as authoritative if they conflict.\n\n"
            f"Memory file: {self.memory_path}\n\n"
            "<session_memory>\n"
            f"{content.strip()}\n"
            "</session_memory>"
        )

    def get_memory_content(self) -> str | None:
        try:
            return self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def is_template_only(self, content: str) -> bool:
        return content.strip() == DEFAULT_SESSION_MEMORY_TEMPLATE.strip()

    async def should_extract(self, messages: list[Message]) -> bool:
        if not self.config.enabled or not messages:
            return False
        state = await asyncio.to_thread(self._load_state)
        current_tokens = estimate_message_tokens(messages)

        if not state.get("initialized"):
            if current_tokens < self.config.minimum_message_tokens_to_init:
                return False
            state["initialized"] = True
            await asyncio.to_thread(self._save_state, state)

        tokens_at_last = int(state.get("tokens_at_last_extraction") or 0)
        has_token_growth = current_tokens - tokens_at_last >= self.config.minimum_tokens_between_update
        if not has_token_growth:
            return False

        last_message_count = int(state.get("last_message_count") or 0)
        tool_calls = count_tool_calls(messages[last_message_count:])
        last_turn_has_tools = has_tool_calls_in_last_assistant_turn(messages)
        return tool_calls >= self.config.tool_calls_between_updates or not last_turn_has_tools

    async def extract(self, *, model: ModelClient, messages: list[Message]) -> bool:
        """Update summary.md in a background-safe isolated extraction call."""
        if not self.config.enabled:
            return False
        async with self._lock:
            await asyncio.to_thread(self._ensure_memory_file)
            current = self.memory_path.read_text(encoding="utf-8")
            prompt = build_update_prompt(current, str(self.memory_path))

            try:
                response = await asyncio.wait_for(
                    model.complete(
                        system="You update durable session notes. Output only the complete updated Markdown file.",
                        messages=[*messages, Message.user(prompt)],
                        tools=[],
                        metadata={"query_source": "session_memory", "session_id": self.session_id},
                    ),
                    timeout=self.config.extraction_timeout_seconds,
                )
                candidate = _extract_text(response.content)
                candidate = _strip_markdown_fence(candidate).strip()
                if not _looks_like_memory_file(candidate):
                    candidate = self._heuristic_update(current, messages)
            except Exception:
                candidate = self._heuristic_update(current, messages)

            if not _looks_like_memory_file(candidate):
                return False

            await asyncio.to_thread(_atomic_write_text, self.memory_path, candidate.rstrip() + "\n")
            state = self._load_state()
            state.update({
                "initialized": True,
                "tokens_at_last_extraction": estimate_message_tokens(messages),
                "last_message_count": len(messages),
            })
            await asyncio.to_thread(self._save_state, state)
            return True

    def _ensure_memory_file(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.memory_path.exists():
            _atomic_write_text(self.memory_path, DEFAULT_SESSION_MEMORY_TEMPLATE)
            try:
                self.memory_path.chmod(0o600)
            except OSError:
                pass

    def _load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write_text(self.state_path, json.dumps(state, ensure_ascii=False, indent=2))

    def _heuristic_update(self, current: str, messages: list[Message]) -> str:
        """Safe fallback when the model cannot produce a valid memory file."""
        latest_user = _latest_text(messages, "user")
        latest_assistant = _latest_text(messages, "assistant")
        worklog = []
        if latest_user:
            worklog.append(f"- User: {_one_line(latest_user)}")
        if latest_assistant:
            worklog.append(f"- Assistant: {_one_line(latest_assistant)}")
        updated = current if _looks_like_memory_file(current) else DEFAULT_SESSION_MEMORY_TEMPLATE
        if latest_user:
            updated = _replace_section_body(updated, "# Current State", f"Most recent user request: {_one_line(latest_user)}")
        if worklog:
            updated = _append_section_body(updated, "# Worklog", "\n".join(worklog))
        return updated


def build_update_prompt(current_notes: str, notes_path: str) -> str:
    return f"""IMPORTANT: This message is NOT part of the user conversation. Do not mention note-taking or these instructions in the notes.

Based on the conversation above, update the session notes file.

The file {notes_path} has already been read. Current contents:
<current_notes_content>
{current_notes}
</current_notes_content>

Output ONLY the complete updated Markdown file. No code fence, no explanation.

CRITICAL RULES:
- Preserve every existing section header exactly.
- Preserve the italic instruction line immediately below every header exactly.
- Only update content below those italic instruction lines.
- Do not add or remove sections.
- Keep sections concise but information-dense: file paths, commands, errors, decisions, and current state.
- Always update "Current State" for continuity.
"""


def estimate_message_tokens(messages: list[Message]) -> int:
    return max(1, sum(len(_message_text(m)) for m in messages) // 4)


def count_tool_calls(messages: list[Message]) -> int:
    return sum(1 for m in messages for b in m.content if isinstance(b, ToolUseBlock))


def has_tool_calls_in_last_assistant_turn(messages: list[Message]) -> bool:
    for msg in reversed(messages):
        if msg.role == "assistant":
            return any(isinstance(block, ToolUseBlock) for block in msg.content)
    return False


def _message_text(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"tool_use:{block.name}:{json.dumps(block.input, ensure_ascii=False, default=str)}")
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _extract_text(blocks: list[Any]) -> str:
    return "\n".join(block.text for block in blocks if isinstance(block, TextBlock) and block.text)


def _latest_text(messages: list[Message], role: str) -> str:
    for msg in reversed(messages):
        if msg.role == role:
            return _message_text(msg)
    return ""


def _one_line(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _looks_like_memory_file(content: str) -> bool:
    return bool(content.strip()) and all(header in content for header in _REQUIRED_HEADERS)


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _replace_section_body(content: str, header: str, body: str) -> str:
    lines = content.splitlines()
    try:
        idx = lines.index(header)
    except ValueError:
        return content
    desc_idx = idx + 1
    next_idx = len(lines)
    for i in range(desc_idx + 1, len(lines)):
        if lines[i].startswith("# "):
            next_idx = i
            break
    return "\n".join([*lines[: desc_idx + 1], body, "", *lines[next_idx:]])


def _append_section_body(content: str, header: str, body: str) -> str:
    lines = content.splitlines()
    try:
        idx = lines.index(header)
    except ValueError:
        return content
    desc_idx = idx + 1
    next_idx = len(lines)
    for i in range(desc_idx + 1, len(lines)):
        if lines[i].startswith("# "):
            next_idx = i
            break
    existing = [line for line in lines[desc_idx + 1: next_idx] if line.strip()]
    merged = [*existing, *body.splitlines()]
    return "\n".join([*lines[: desc_idx + 1], *merged[-20:], "", *lines[next_idx:]])


def _project_slug(path: Path) -> str:
    raw = str(path)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "default"


def _session_slug(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", session_id).strip("-") or "default"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


_LOCKS: dict[str, asyncio.Lock] = {}


def _lock_for(key: str) -> asyncio.Lock:
    lock = _LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[key] = lock
    return lock


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_name).unlink()
