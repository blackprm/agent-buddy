from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.permissions.policy import StaticPermissionPolicy, get_session_permission_state
from agent_core.plan_mode import get_plan, get_plan_file_path, prepare_context_for_plan_mode, write_plan
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolContext, ToolRegistry
from agent_core.tools.builtin import BashTool, EnterPlanModeTool, ExitPlanModeTool, WriteTextFileTool
from agent_core.types import TextBlock, ToolUseBlock


@pytest.mark.asyncio
async def test_enter_plan_mode_sets_mode_and_returns_plan_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-enter-test"
    context = ToolContext(session_id=session_id, messages=[], cwd=str(tmp_path))

    result = await EnterPlanModeTool().call({}, context)

    assert result.is_error is False
    assert get_session_permission_state(session_id).mode == "plan"
    assert result.metadata["previousMode"] == "default"
    assert str(tmp_path / "plans") in result.metadata["planFilePath"]


@pytest.mark.asyncio
async def test_plan_mode_allows_only_plan_file_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-policy-test"
    prepare_context_for_plan_mode(session_id)
    policy = StaticPermissionPolicy(session_id=session_id, cwd=tmp_path)

    project_write = await policy.check(tool=WriteTextFileTool(), tool_input={"path": "src/app.py", "content": "x"})
    plan_write = await policy.check(tool=WriteTextFileTool(), tool_input={"path": str(get_plan_file_path(session_id, cwd=tmp_path)), "content": "plan"})
    bash = await policy.check(tool=BashTool(), tool_input={"command": "pwd"})

    assert project_write.status == "deny"
    assert plan_write.status == "allow"
    assert bash.status == "deny"


@pytest.mark.asyncio
async def test_exit_plan_mode_approval_restores_selected_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-exit-test"
    prepare_context_for_plan_mode(session_id)
    write_plan(session_id, "# Plan\n\n1. Do the thing.\n", cwd=tmp_path)
    policy = StaticPermissionPolicy(session_id=session_id, cwd=tmp_path)
    tool = ExitPlanModeTool()

    decision = await policy.check(tool=tool, tool_input={})
    assert decision.status == "ask"
    assert decision.metadata["kind"] == "plan_approval"
    assert "Do the thing" in decision.metadata["plan"]

    option = next(o for o in decision.options if o.get("mode") == "acceptEdits")
    policy.record_user_decision(tool=tool, tool_input={}, option=option)

    result = await tool.call({}, ToolContext(session_id=session_id, messages=[], cwd=str(tmp_path)))

    assert result.is_error is False
    assert get_session_permission_state(session_id).mode == "acceptEdits"
    assert "Plan approved" in result.content
    assert "Do the thing" in result.metadata["plan"]


@pytest.mark.asyncio
async def test_runtime_starts_implementation_after_plan_approval_clear_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-handoff-clear-test"
    prepare_context_for_plan_mode(session_id)
    write_plan(session_id, "# Plan\n\n1. Implement the approved change.\n", cwd=tmp_path)

    async def approve(_tool_name: str, _tool_input: dict, decision) -> dict:
        option = next(o for o in decision.options if o["value"] == "yes-restore-clear-context")
        return {"decision": "allow", "option": option}

    runtime = AgentRuntime(
        model=ScriptedModelClient([
            ModelResponse(content=[ToolUseBlock(id="toolu_plan", name="ExitPlanMode", input={})], stop_reason="tool_use"),
            ModelResponse(content=[TextBlock("implementation started")], stop_reason="end_turn"),
        ]),
        tools=ToolRegistry([ExitPlanModeTool()]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path), max_turns=5),
        ask_callback=approve,
    )

    events = [event async for event in runtime.run("please plan first")]

    assert any(event.type == "plan_implementation_started" and event.data["clearContext"] is True for event in events)
    assert any(event.type == "assistant_text" and "implementation started" in event.data["text"] for event in events)
    assert runtime.messages[0].role == "user"
    assert isinstance(runtime.messages[0].content[0], TextBlock)
    assert "Implement the following approved plan" in runtime.messages[0].content[0].text
    assert "please plan first" not in runtime.messages[0].content[0].text


@pytest.mark.asyncio
async def test_runtime_starts_implementation_after_plan_approval_keep_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-handoff-keep-test"
    prepare_context_for_plan_mode(session_id)
    write_plan(session_id, "# Plan\n\n1. Keep context while implementing.\n", cwd=tmp_path)

    async def approve(_tool_name: str, _tool_input: dict, decision) -> dict:
        option = next(o for o in decision.options if o["value"] == "yes-restore-keep-context")
        return {"decision": "allow", "option": option}

    runtime = AgentRuntime(
        model=ScriptedModelClient([
            ModelResponse(content=[ToolUseBlock(id="toolu_plan", name="ExitPlanMode", input={})], stop_reason="tool_use"),
            ModelResponse(content=[TextBlock("implementation continued")], stop_reason="end_turn"),
        ]),
        tools=ToolRegistry([ExitPlanModeTool()]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path), max_turns=5),
        ask_callback=approve,
    )

    events = [event async for event in runtime.run("keep this context")]

    assert any(event.type == "plan_implementation_started" and event.data["clearContext"] is False for event in events)
    assert any(event.type == "assistant_text" and "implementation continued" in event.data["text"] for event in events)
    user_texts = [block.text for message in runtime.messages for block in message.content if isinstance(block, TextBlock) and message.role == "user"]
    assert any("keep this context" in text for text in user_texts)
    assert any("Implement the following approved plan" in text for text in user_texts)


@pytest.mark.asyncio
async def test_runtime_persists_and_restores_plan_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-persist-restore-test"
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    runtime = AgentRuntime(
        model=ScriptedModelClient([
            ModelResponse(content=[ToolUseBlock(id="toolu_enter", name="EnterPlanMode", input={})], stop_reason="tool_use"),
            ModelResponse(content=[TextBlock("planned")], stop_reason="end_turn"),
        ]),
        tools=ToolRegistry([EnterPlanModeTool()]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path), max_turns=5),
        session_store=store,
    )

    _ = [event async for event in runtime.run("enter plan mode")]
    saved_meta = store.get_session(session_id)["metadata"]["plan_mode"]
    saved_slug = saved_meta["plan_slug"]
    assert saved_meta["mode"] == "plan"
    assert saved_slug

    state = get_session_permission_state(session_id)
    state.mode = "default"
    state.pre_plan_mode = ""
    state.plan_slug = ""

    AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("noop")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path)),
        session_store=store,
    )

    restored = get_session_permission_state(session_id)
    assert restored.mode == "plan"
    assert restored.plan_slug == saved_slug


@pytest.mark.asyncio
async def test_runtime_recovers_missing_plan_file_from_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PLANS_DIR", str(tmp_path / "plans"))
    session_id = "plan-file-recovery-test"
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    prepare_context_for_plan_mode(session_id)
    write_plan(session_id, "# Plan\n\n1. Recover this plan.\n", cwd=tmp_path)

    async def approve(_tool_name: str, _tool_input: dict, decision) -> dict:
        option = next(o for o in decision.options if o["value"] == "yes-restore-clear-context")
        return {"decision": "allow", "option": option}

    runtime = AgentRuntime(
        model=ScriptedModelClient([
            ModelResponse(content=[ToolUseBlock(id="toolu_plan", name="ExitPlanMode", input={})], stop_reason="tool_use"),
            ModelResponse(content=[TextBlock("implemented")], stop_reason="end_turn"),
        ]),
        tools=ToolRegistry([ExitPlanModeTool()]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path), max_turns=5),
        session_store=store,
        ask_callback=approve,
    )
    _ = [event async for event in runtime.run("approve plan")]
    plan_path = get_plan_file_path(session_id, cwd=tmp_path)
    assert plan_path.exists()
    plan_path.unlink()

    state = get_session_permission_state(session_id)
    state.mode = "default"
    state.pre_plan_mode = ""
    state.plan_slug = ""

    AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("noop")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path)),
        session_store=store,
    )

    assert plan_path.exists()
    assert "Recover this plan" in (get_plan(session_id, cwd=tmp_path) or "")
