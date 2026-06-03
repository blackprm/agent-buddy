from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_core.types import ToolUseBlock


MAX_ERROR_CHARS = 10_000


class ToolFailureCategory(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INPUT_VALIDATION = "input_validation"
    PERMISSION_DENIED = "permission_denied"
    HOOK_BLOCKED = "hook_blocked"
    USER_DENIED = "user_denied"
    BASH_EXIT = "bash_exit"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    FILE_NOT_FOUND = "file_not_found"
    TOOL_EXCEPTION = "tool_exception"


@dataclass(slots=True)
class ToolFailure:
    category: ToolFailureCategory
    message: str
    hint: str = ""
    retryable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "failureCategory": self.category.value,
            "recoveryHint": self.hint,
            "retryable": self.retryable,
            **self.metadata,
        }


def _truncate(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [{len(text) - limit} characters truncated] ...\n\n{text[-half:]}"


def _input_keys(tool_use: ToolUseBlock) -> str:
    keys = sorted((tool_use.input or {}).keys())
    return ", ".join(keys) if keys else "none"


def format_exception_detail(exc: BaseException) -> str:
    """Return the full exception payload that should be visible to the agent.

    Tool wrappers often catch provider/client exceptions and convert them into
    ToolResult(is_error=True).  If they only include ``str(exc)``, the next model
    turn loses status codes, stderr/stdout, response bodies, chained exceptions,
    and the traceback location.  Keep those details in the tool result content so
    the agent can diagnose and choose a different recovery path.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    code = getattr(exc, "returncode", getattr(exc, "code", None))
    if code is not None:
        parts.append(f"Exit code: {code}")
    for attr, label in (("stderr", "STDERR"), ("stdout", "STDOUT")):
        value = getattr(exc, attr, "")
        if value:
            parts.append(f"{label}:\n{value}")
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        text = getattr(response, "text", "")
        if status is not None:
            parts.append(f"HTTP status: {status}")
        if text:
            parts.append(f"HTTP response body:\n{text}")
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if traceback_text:
        parts.append(f"Traceback:\n{traceback_text}")
    return _truncate("\n".join(str(part) for part in parts if part))


def classify_tool_failure(
    *,
    tool_name: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ToolFailure:
    metadata = metadata or {}
    text = content or ""
    lower = text.lower()
    if tool_name == "bash" or "exit code" in lower or metadata.get("exitCode") not in (None, 0):
        code = metadata.get("exitCode")
        hint = "Read the exit code, stderr, and command output before retrying. Change the command or inspect the referenced files first."
        return ToolFailure(ToolFailureCategory.BASH_EXIT, text, hint, retryable=True, metadata={"exitCode": code})
    if "missing required input field" in lower or "inputvalidationerror" in lower:
        return ToolFailure(
            ToolFailureCategory.INPUT_VALIDATION,
            text,
            "Retry the same tool with the required field names and correct JSON value types. If this was a deferred tool, activate it with ToolSearch first.",
            retryable=True,
        )
    if "unknown tool" in lower or "no such tool" in lower:
        return ToolFailure(
            ToolFailureCategory.UNKNOWN_TOOL,
            text,
            "Use ToolSearch to discover or activate the correct tool name before retrying.",
            retryable=True,
        )
    if "hook blocked" in lower or "blocked by hook" in lower or "hook denied" in lower:
        return ToolFailure(
            ToolFailureCategory.HOOK_BLOCKED,
            text,
            "Do not repeat the same blocked action. Address the hook feedback or choose a safer alternative.",
            retryable=False,
        )
    if "user denied" in lower or "permission denied" in lower or "permission required" in lower:
        return ToolFailure(
            ToolFailureCategory.PERMISSION_DENIED,
            text,
            "The operation was not executed. Do not try to bypass the denial; explain the need or choose a clearly allowed alternative.",
            retryable=False,
        )
    if "timed out" in lower or "timeout" in lower:
        return ToolFailure(
            ToolFailureCategory.TIMEOUT,
            text,
            "Check whether the command is still running in the background, inspect partial output, or retry with a narrower command/longer timeout.",
            retryable=True,
        )
    if "interrupted" in lower or "aborted" in lower:
        return ToolFailure(
            ToolFailureCategory.INTERRUPTED,
            text,
            "The operation was interrupted. Wait for explicit user direction before repeating destructive or long-running work.",
            retryable=False,
        )
    if "no such file" in lower or "not found" in lower or "does not exist" in lower:
        return ToolFailure(
            ToolFailureCategory.FILE_NOT_FOUND,
            text,
            "Verify the path with Glob/ListDirectory before retrying the file operation.",
            retryable=True,
        )
    return ToolFailure(
        ToolFailureCategory.TOOL_EXCEPTION,
        text,
        "Diagnose the error before retrying. Avoid repeating the identical tool call with the same input.",
        retryable=True,
    )


def format_tool_exception(tool_use: ToolUseBlock, exc: BaseException) -> ToolFailure:
    keys_text = _input_keys(tool_use)
    if isinstance(exc, KeyError):
        missing = exc.args[0] if exc.args else "unknown"
        message = (
            f"Tool {tool_use.name} failed: missing required input field {missing!r}. "
            f"Received input keys: {keys_text}. Please retry with the required field."
        )
        return ToolFailure(
            ToolFailureCategory.INPUT_VALIDATION,
            message,
            "Use the tool schema field names exactly; do not invent aliases unless the schema documents them.",
            retryable=True,
            metadata={"missingField": str(missing), "receivedKeys": sorted((tool_use.input or {}).keys())},
        )
    code = getattr(exc, "returncode", getattr(exc, "code", None))
    message = _truncate(f"Tool {tool_use.name} failed.\n{format_exception_detail(exc)}\nReceived input keys: {keys_text}.")
    return classify_tool_failure(tool_name=tool_use.name, content=message, metadata={"exitCode": code})


def recovery_hint_for_tool_result(tool_name: str, content: str, metadata: dict[str, Any] | None = None) -> ToolFailure:
    failure = classify_tool_failure(tool_name=tool_name, content=content, metadata=metadata)
    if tool_name.startswith("Task") and failure.category != ToolFailureCategory.INPUT_VALIDATION:
        failure.metadata["taskFailure"] = True
    return failure


def repeated_failure_hint(tool_name: str, content: str, *, count: int) -> str:
    normalized = re.sub(r"\s+", " ", content.strip())[:240]
    return (
        f"Repeated tool failure detected ({count}x) for {tool_name}: {normalized}. "
        "Do not repeat the identical call. Change strategy, inspect prerequisites, or ask for permission/context if needed."
    )
