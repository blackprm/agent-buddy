from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.permissions.policy import FilePermissionRule, StaticPermissionPolicy, get_session_permission_state, reset_session_permission_rules
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.builtin import BashTool, WriteTextFileTool
from agent_core.types import TextBlock
from agent_server.slash_commands import handle_slash_command


@pytest.mark.asyncio
async def test_runtime_persists_and_restores_permission_rules(tmp_path: Path) -> None:
    session_id = "permission-persist-restore-test"
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    state = get_session_permission_state(session_id)
    state.mode = "acceptEdits"
    state.bash_prefixes.add("npm run")
    state.skill_rules.add("frontend-design")
    state.file_rules.append(FilePermissionRule(root=str(tmp_path / "src"), operation="write", scope="manual"))

    runtime = AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path), max_turns=2),
        session_store=store,
    )
    _ = [event async for event in runtime.run("persist permissions")]

    saved = store.get_session(session_id)["metadata"]["permissions"]
    assert saved["mode"] == "acceptEdits"
    assert saved["bash_prefixes"] == ["npm run"]
    assert saved["skill_rules"] == ["frontend-design"]
    assert saved["file_rules"][0]["root"] == str(tmp_path / "src")

    reset_session_permission_rules(session_id, reset_mode=True)
    restored_before = get_session_permission_state(session_id)
    assert restored_before.mode == "default"
    assert restored_before.bash_prefixes == set()

    AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("noop")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        permission_policy=StaticPermissionPolicy(session_id=session_id, cwd=tmp_path),
        config=AgentRuntimeConfig(session_id=session_id, cwd=str(tmp_path)),
        session_store=store,
    )

    restored = get_session_permission_state(session_id)
    assert restored.mode == "acceptEdits"
    assert restored.bash_prefixes == {"npm run"}
    assert restored.skill_rules == {"frontend-design"}
    assert restored.file_rules[0].root == str(tmp_path / "src")


@pytest.mark.asyncio
async def test_permissions_command_manages_persisted_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    session_id = "permissions-command-test"
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")

    async def not_running(_session_id: str) -> bool:
        return False

    async def abort(_session_id: str) -> bool:
        return True

    bash_result = await handle_slash_command(
        "/permissions allow-bash npm run",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    skill_result = await handle_slash_command(
        "/permissions allow-skill /frontend-design",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    write_result = await handle_slash_command(
        "/permissions allow-write src",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )

    assert bash_result["permission_rules"]["bash_prefixes"] == ["npm run"]
    assert skill_result["permission_rules"]["skill_rules"] == ["frontend-design"]
    assert write_result["permission_rules"]["file_rules"][0]["root"] == str(tmp_path / "src")
    saved = store.get_session(session_id)["metadata"]["permissions"]
    assert saved["bash_prefixes"] == ["npm run"]
    assert saved["skill_rules"] == ["frontend-design"]

    policy = StaticPermissionPolicy(session_id=session_id, cwd=tmp_path)
    bash_decision = await policy.check(tool=BashTool(), tool_input={"command": "npm run build"})
    write_decision = await policy.check(tool=WriteTextFileTool(), tool_input={"path": "src/app.py", "content": "x"})
    assert bash_decision.status == "allow"
    assert write_decision.status == "allow"

    revoke_result = await handle_slash_command(
        "/permissions revoke-bash npm run",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    assert revoke_result["permission_rules"]["bash_prefixes"] == []

    reset_result = await handle_slash_command(
        "/permissions reset",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    assert reset_result["permission_mode"] == "default"
    assert reset_result["permission_rules"] == {
        "bash_prefixes": [],
        "skill_rules": [],
        "file_rules": [],
        "web_domains": [],
        "web_search_allowed": False,
    }
