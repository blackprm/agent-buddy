from __future__ import annotations

from pathlib import Path

from agent_core.context.builder import ContextBuilder
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.permissions.policy import StaticPermissionPolicy
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.teams import AgentTool, ReadInboxTool, SendMessageTool, TeamCreateTool, TeamListTool, TeamStore
from agent_core.tools.base import ToolContext, ToolRegistry
from agent_core.types import TextBlock


def test_team_store_sends_broadcasts_and_drains_inboxes(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    team = store.create_team(name="migration-team", lead_session_id="lead-session", description="Migrate agents")
    store.upsert_member(
        team_name=team.name,
        member_name="alice",
        role="coder",
        child_session_id="lead-session:team:migration-team:alice",
    )
    store.upsert_member(
        team_name=team.name,
        member_name="bob",
        role="reviewer",
        child_session_id="lead-session:team:migration-team:bob",
    )

    sent = store.send_message(team_name=team.name, sender="team-lead", recipient="*", content="hello")
    assert sent == ["alice", "bob"]

    alice_messages = store.read_inbox(team_name=team.name, recipient="alice")
    assert len(alice_messages) == 1
    assert alice_messages[0].type == "broadcast"
    assert alice_messages[0].sender == "team-lead"
    assert store.read_inbox(team_name=team.name, recipient="alice") == []


async def test_team_tools_create_list_send_and_read(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    context = ToolContext(session_id="lead-session", messages=[], metadata={"user_id": "u1", "org_id": "o1"})

    created = await TeamCreateTool(store).call({"team_name": "migration-team", "description": "Migrate agents"}, context)
    assert created.is_error is False
    assert "migration-team" in created.content

    listed = await TeamListTool(store).call({}, context)
    assert "Team `migration-team`" in listed.content

    store.upsert_member(
        team_name="migration-team",
        member_name="alice",
        role="coder",
        child_session_id="lead-session:team:migration-team:alice",
    )
    sent = await SendMessageTool(store).call(
        {"team_name": "migration-team", "to": "alice", "content": "please implement", "summary": "assignment"},
        context,
    )
    assert sent.is_error is False
    assert "Sent message to alice" in sent.content

    inbox = await ReadInboxTool(store).call({"team_name": "migration-team", "recipient": "alice"}, context)
    assert inbox.is_error is False
    assert "please implement" in inbox.content
    assert inbox.metadata["messages"][0]["summary"] == "assignment"


async def test_agent_tool_persists_named_teammate_session_and_result(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    session_store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    model = ScriptedModelClient([ModelResponse(content=[TextBlock("teammate done")], stop_reason="end_turn")])

    def context_builder_factory(child_session_id: str, subagent_type: str | None = None) -> ContextBuilder:
        return ContextBuilder(product_name="Test", base_instructions=f"Child {child_session_id} {subagent_type}")

    def tools_factory() -> ToolRegistry:
        return ToolRegistry([SendMessageTool(store), ReadInboxTool(store)])

    tool = AgentTool(
        team_store=store,
        model=model,
        sub_tools_factory=tools_factory,
        context_builder_factory=context_builder_factory,
        permission_policy_factory=lambda: StaticPermissionPolicy(allow={"SendMessage", "ReadInbox"}),
        session_store=session_store,
        max_turns=3,
    )
    context = ToolContext(
        session_id="lead-session",
        messages=[],
        metadata={"user_id": "u1", "org_id": "o1"},
        cwd=str(tmp_path),
    )

    result = await tool.call(
        {
            "description": "implement thing",
            "prompt": "Do the assigned work",
            "team_name": "migration-team",
            "name": "alice",
            "subagent_type": "coder",
        },
        context,
    )

    assert result.is_error is False
    assert result.content == "teammate done"
    assert result.metadata["teamName"] == "migration-team"
    assert result.metadata["agentName"] == "alice"

    team = store.require_team("migration-team")
    member = team.find_member("alice")
    assert member is not None
    assert member.status == "idle"
    assert member.last_result == "teammate done"
    assert session_store.get_session(member.child_session_id) is not None

    lead_messages = store.read_inbox(team_name="migration-team", recipient="team-lead")
    assert lead_messages[0].content == "teammate done"
