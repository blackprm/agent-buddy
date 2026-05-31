from __future__ import annotations

from collections import deque
from typing import Any

from agent_core.model.base import ModelClient, ModelResponse
from agent_core.types import Message, SystemPrompt


class ScriptedModelClient(ModelClient):
    """用于本地验证/单测的脚本化模型。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)

    async def complete(
        self,
        *,
        system: str | SystemPrompt,
        messages: list[Message],
        tools: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self._responses:
            return ModelResponse(content=[], stop_reason="end_turn")
        return self._responses.popleft()
