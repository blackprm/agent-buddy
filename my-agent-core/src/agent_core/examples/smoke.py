from __future__ import annotations

import asyncio

from agent_core.core.agent import AgentRuntime
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.tools.base import ToolRegistry
from agent_core.tools.builtin import EchoTool
from agent_core.types import TextBlock, ToolUseBlock


async def main() -> None:
    model = ScriptedModelClient(
        [
            ModelResponse(
                content=[
                    TextBlock("我先调用 echo 工具。"),
                    ToolUseBlock(id="toolu_1", name="echo", input={"text": "hello agent"}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(content=[TextBlock("工具返回后，任务完成。")], stop_reason="end_turn"),
        ]
    )
    runtime = AgentRuntime(model=model, tools=ToolRegistry([EchoTool()]))
    async for event in runtime.run("跑一个 smoke test"):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
