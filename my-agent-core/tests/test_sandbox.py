from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.permissions.policy import StaticPermissionPolicy
from agent_core.sandbox.manager import SandboxManager


def _darwin_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SandboxManager:
    monkeypatch.setattr("agent_core.sandbox.manager.platform.system", lambda: "Darwin")
    monkeypatch.setattr("agent_core.sandbox.manager.shutil.which", lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None)
    return SandboxManager(cwd=tmp_path, config_path=tmp_path / "sandbox.json")


def test_should_use_sandbox_respects_enabled_dangerous_disable_and_closed_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _darwin_manager(tmp_path, monkeypatch)
    manager.set_settings(enabled=True, allow_unsandboxed_commands=True)

    assert manager.should_use_sandbox({"command": "python script.py"}) is True
    assert manager.should_use_sandbox({"command": "python script.py", "dangerously_disable_sandbox": True}) is False

    manager.set_settings(allow_unsandboxed_commands=False)
    assert manager.should_use_sandbox({"command": "python script.py", "dangerouslyDisableSandbox": True}) is True


def test_excluded_commands_match_compound_commands_and_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _darwin_manager(tmp_path, monkeypatch)
    manager.set_settings(enabled=True, excluded_commands=["git:*", "docker ps", "npm test"])

    assert manager.contains_excluded_command("echo ok && git status") is True
    assert manager.contains_excluded_command("timeout 30 npm test -- --watch=false") is True
    assert manager.contains_excluded_command("docker ps") is True
    assert manager.contains_excluded_command("python -m pytest") is False


def test_darwin_wrapper_contains_filesystem_and_network_restrictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _darwin_manager(tmp_path, monkeypatch)
    manager.set_settings(
        enabled=True,
        network={"allowed_domains": ["example.com"]},
        filesystem={"allow_write": ["."], "deny_write": ["/etc"], "deny_read": ["~/.ssh"], "allow_read": []},
    )

    wrapped = manager.wrap_command("echo hello", cwd=tmp_path)

    assert "sandbox-exec" in wrapped
    assert "deny network" in wrapped
    assert "deny file-write" in wrapped
    assert "allow file-write" in wrapped
    assert "echo hello" in wrapped


@pytest.mark.asyncio
async def test_permission_policy_auto_allows_sandboxed_bash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _darwin_manager(tmp_path, monkeypatch)
    manager.set_settings(enabled=True, auto_allow_bash_if_sandboxed=True)
    monkeypatch.setattr("agent_core.permissions.policy.get_sandbox_manager", lambda: manager)

    class Tool:
        name = "bash"

    policy = StaticPermissionPolicy(session_id="sandbox-test", cwd=tmp_path)
    decision = await policy.check(tool=Tool(), tool_input={"command": "python script.py"})

    assert decision.status == "allow"
    assert decision.metadata["sandboxed"] is True
