from __future__ import annotations

import subprocess
from pathlib import Path

from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.tools.base import ToolContext, ToolRegistry
from agent_core.tools.builtin import BashTool, ReadTextFileTool, WriteTextFileTool
from agent_core.types import TextBlock
from agent_core.worktree import WorktreeManager, validate_worktree_slug, worktree_branch_name


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("main\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")


def test_worktree_slug_validation() -> None:
    validate_worktree_slug("task-session-1")
    assert worktree_branch_name("task/session") == "worktree-task+session"
    for invalid in ("../x", "/abs", "has space", "a" * 65):
        try:
            validate_worktree_slug(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid slug: {invalid}")


async def test_worktree_manager_creates_preserves_and_removes_clean_workspace(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    manager = WorktreeManager(cwd=repo)
    info = await manager.create_agent_worktree("task-s1-1", session_id="s1")
    worktree = Path(info.worktree_path)

    assert worktree.exists()
    assert (worktree / "README.md").read_text(encoding="utf-8") == "main\n"
    assert info.worktree_branch == "worktree-task-s1-1"

    removed, changed = await manager.cleanup_if_clean(info)
    assert changed is False
    assert removed is True
    assert not worktree.exists()


async def test_worktree_manager_preserves_changed_workspace(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    manager = WorktreeManager(cwd=repo)
    info = await manager.create_agent_worktree("task-s1-2", session_id="s1")
    worktree = Path(info.worktree_path)
    (worktree / "README.md").write_text("changed\n", encoding="utf-8")

    removed, changed = await manager.cleanup_if_clean(info)
    assert changed is True
    assert removed is False
    assert worktree.exists()

    await manager.remove_agent_worktree(info)


async def test_tool_context_resolves_relative_paths_from_runtime_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = ToolContext(session_id="s1", messages=[], cwd=str(workspace))

    wrote = await WriteTextFileTool().call({"path": "notes/todo.txt", "content": "hello"}, context)
    assert wrote.is_error is False
    assert (workspace / "notes" / "todo.txt").read_text(encoding="utf-8") == "hello"

    read = await ReadTextFileTool().call({"path": "notes/todo.txt"}, context)
    assert read.content == "hello"

    bash = await BashTool().call({"command": "pwd", "timeout": 5000}, context)
    assert str(workspace) in bash.content


async def test_runtime_loop_started_exposes_workspace_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("done")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        config=AgentRuntimeConfig(session_id="s1", cwd=str(workspace), metadata={"workspace": {"status": "active"}}),
    )

    events = []
    async for event in runtime.run("hi"):
        events.append(event)
    started = next(event for event in events if event.type == "loop_started")
    assert started.data["cwd"] == str(workspace)
    assert started.data["workspace"]["status"] == "active"
