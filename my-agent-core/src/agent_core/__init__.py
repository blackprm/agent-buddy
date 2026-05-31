"""通用 Agent 底座。

设计目标：
- 后端优先：核心 loop 是 async generator，天然适合 HTTP SSE / WebSocket。
- 模型无关：通过 ModelClient 协议适配 Anthropic/OpenAI/内部网关。
- 工具无关：Tool 是 Strategy，ToolRegistry 是 Registry，PermissionPolicy 是 Chain/Policy。
"""

from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.core.events import AgentEvent
from agent_core.model.base import ModelClient, ModelResponse, StreamDelta
from agent_core.types import (
    Message,
    SystemPrompt,
    SystemPromptBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    as_system_prompt,
    system_prompt_to_str,
    system_prompt_to_blocks,
)

__all__ = [
    "AgentEvent",
    "AgentRuntime",
    "AgentRuntimeConfig",
    "Message",
    "ModelClient",
    "ModelResponse",
    "StreamDelta",
    "SystemPrompt",
    "SystemPromptBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "as_system_prompt",
    "system_prompt_to_str",
    "system_prompt_to_blocks",
]
