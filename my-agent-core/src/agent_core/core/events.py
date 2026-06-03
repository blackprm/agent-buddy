from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AgentEventType = Literal[
    "loop_started",
    "turn_started",
    "model_started",
    "model_completed",
    "text_delta",
    "thinking_delta",
    "assistant_text",
    "tool_started",
    "tool_progress",
    "tool_completed",
    "tool_denied",
    "permission_request",
    "permission_result",
    "plan_implementation_started",
    "team_state",
    "agent_status",
    "context_compacting",
    "context_compacted",
    "context_compact_failed",
    "hook_message",
    "loop_completed",
    "loop_failed",
    "loop_aborted",
]


@dataclass(slots=True)
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)
