from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.billing.store import BillingStore
from agent_core.buddy import build_companion_prompt, companion_payload, get_companion, roll
from agent_core.buddy.store import BuddyStore
from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.quota.store import QuotaStore
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolRegistry
from agent_core.types import TextBlock
from agent_core.users.store import UserStore


def test_default_user_context_contains_account_and_organization_uuid(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")

    ctx = store.get_user_context()

    assert ctx["user"]["id"] == "local-user"
    assert ctx["user"]["account_uuid"]
    assert ctx["organization"]["id"] == "local-org"
    assert ctx["organization"]["organization_uuid"]
    assert ctx["role"] == "owner"


def test_user_password_is_hashed_and_verifiable(tmp_path: Path) -> None:
    store = UserStore(tmp_path / "users.db")
    user = store.upsert_user({"id": "alice", "name": "Alice", "password": "secret123"})

    assert user["has_password"] is True
    assert "password_hash" not in user
    assert store.verify_password(user_id="alice", password="secret123")["id"] == "alice"
    assert store.verify_password(user_id="alice", password="wrong") is None


def test_buddy_roll_is_stable_and_payload_contains_web_sprite() -> None:
    first = roll("account-alice")
    second = roll("account-alice")

    assert first == second
    payload = companion_payload(get_companion({
        "user": {"id": "missing", "account_uuid": "account-alice"},
        "organization": {"id": "org"},
    }, create=False))
    assert payload is None


def test_buddy_hatches_per_user_and_prompt_can_be_muted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BUDDY_DB_PATH", str(tmp_path / "buddy.db"))
    ctx = {
        "user": {"id": "alice", "account_uuid": "account-alice"},
        "organization": {"id": "org", "organization_uuid": "org-uuid"},
    }

    companion = get_companion(ctx, create=True)
    payload = companion_payload(companion)

    assert companion.name
    assert payload is not None
    assert payload["sprite_frames"]
    assert companion.species == roll("account-alice").species
    assert companion.rarity == roll("account-alice").rarity
    assert companion.name == get_companion(ctx, create=True).name
    assert "# Companion" in build_companion_prompt(ctx)

    BuddyStore(tmp_path / "buddy.db").set_muted(user_id="alice", muted=True)
    assert build_companion_prompt(ctx) == ""


def test_session_and_billing_keep_user_org_attribution(tmp_path: Path) -> None:
    sessions = SQLiteSessionStore(tmp_path / "sessions.db")
    billing = BillingStore(tmp_path / "billing.db")

    sid = sessions.create_session(session_id="s1", user_id="u1", org_id="o1", metadata={"x": 1})
    info = sessions.get_session(sid)
    billing.record_usage(
        session_id=sid,
        user_id="u1",
        org_id="o1",
        model_id="fake",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )

    assert info is not None
    assert info["user_id"] == "u1"
    assert info["org_id"] == "o1"
    assert billing.session_summary(sid)["total_tokens"] == 15
    assert billing.usage_summary(user_id="u1")["total_tokens"] == 15
    assert billing.usage_summary(org_id="o1")["request_count"] == 1


def test_session_listing_can_be_filtered_by_user(tmp_path: Path) -> None:
    sessions = SQLiteSessionStore(tmp_path / "sessions.db")
    sessions.create_session(session_id="alice-session", user_id="alice", org_id="org")
    sessions.create_session(session_id="bob-session", user_id="bob", org_id="org")

    visible = sessions.list_sessions(user_id="alice")

    assert [session["id"] for session in visible] == ["alice-session"]


def test_runtime_factory_rejects_cross_user_session_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_server import runtime_factory

    users = UserStore(tmp_path / "users.db")
    users.upsert_user({"id": "alice", "name": "Alice"})
    users.upsert_user({"id": "bob", "name": "Bob"})
    users.upsert_organization({"id": "org", "name": "Org"})
    users.add_member(user_id="alice", org_id="org")
    users.add_member(user_id="bob", org_id="org")

    sessions = SQLiteSessionStore(tmp_path / "sessions.db")
    sessions.create_session(session_id="bob-session", user_id="bob", org_id="org")

    monkeypatch.setattr(runtime_factory, "_user_store", users)
    monkeypatch.setattr(runtime_factory, "_session_store", sessions)
    monkeypatch.setattr(runtime_factory, "_billing_store", BillingStore(tmp_path / "billing.db"))
    monkeypatch.setattr(runtime_factory, "_quota_store", QuotaStore(tmp_path / "quota.db"))

    with pytest.raises(PermissionError, match="Session does not belong"):
        runtime_factory.create_runtime(mode="fake", session_id="bob-session", user_id="alice", org_id="org")


def test_quota_preflight_blocks_after_limit_is_reached(tmp_path: Path) -> None:
    quota = QuotaStore(tmp_path / "quota.db")
    quota.upsert_policy({
        "scope_type": "user",
        "scope_id": "u1",
        "name": "Tiny user quota",
        "max_requests_per_day": 1,
    })

    first = quota.check_preflight(user_id="u1", org_id="o1")
    quota.record_usage(user_id="u1", org_id="o1", total_tokens=20, total_cost=0)
    second = quota.check_preflight(user_id="u1", org_id="o1")

    assert first.allowed is True
    assert second.allowed is False
    assert "Quota exceeded" in second.reason


def test_slash_quota_uses_current_user_context_and_returns_progress_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_server.slash_commands import _quota_command

    monkeypatch.setenv("AGENT_QUOTA_DB_PATH", str(tmp_path / "quota.db"))
    monkeypatch.setenv("AGENT_BILLING_DB_PATH", str(tmp_path / "billing.db"))
    monkeypatch.setenv("AGENT_USERS_DB_PATH", str(tmp_path / "users.db"))

    quota = QuotaStore(tmp_path / "quota.db")
    quota.upsert_policy({"scope_type": "user", "scope_id": "alice", "max_requests_per_day": 4})
    quota.record_usage(user_id="alice", org_id="org")

    result = _quota_command(user_context={
        "user": {"id": "alice", "account_uuid": "account-alice"},
        "organization": {"id": "org", "organization_uuid": "org-uuid"},
        "role": "member",
    })

    assert result["type"] == "command_result"
    assert "user: alice" in result["content"]
    assert result["quota_bars"]
    assert result["quota_bars"][0]["percent"] == 25.0
    assert "[█████░░░░░░░░░░░░░░░]" in result["content"]


def test_weekly_quota_window_blocks_after_week_limit(tmp_path: Path) -> None:
    quota = QuotaStore(tmp_path / "quota.db")
    quota.upsert_policy({
        "scope_type": "user",
        "scope_id": "u1",
        "name": "Weekly quota",
        "max_requests_per_week": 2,
    })

    assert quota.check_preflight(user_id="u1", org_id="o1").allowed is True
    quota.record_usage(user_id="u1", org_id="o1")
    assert quota.check_preflight(user_id="u1", org_id="o1").allowed is True
    quota.record_usage(user_id="u1", org_id="o1")
    decision = quota.check_preflight(user_id="u1", org_id="o1")

    assert decision.allowed is False
    assert "week" in decision.reason


async def test_runtime_preflight_blocks_second_request_even_without_usage_tokens(tmp_path: Path) -> None:
    quota = QuotaStore(tmp_path / "quota.db")
    quota.upsert_policy({"scope_type": "user", "scope_id": "u1", "max_requests_per_day": 1})
    runtime = AgentRuntime(
        model=ScriptedModelClient([
            ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn", usage={}),
            ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn", usage={}),
        ]),
        tools=ToolRegistry([]),
        context_builder=ContextBuilder(base_instructions="test"),
        config=AgentRuntimeConfig(session_id="s1", user_id="u1", org_id="o1", quota_store=quota),
    )

    first = [event.type async for event in runtime.run("one")]
    second = [event.type async for event in runtime.run("two")]

    assert "loop_completed" in first
    assert "quota_exceeded" in second
    assert "model_started" not in second


async def test_runtime_emits_companion_reaction_after_clean_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BUDDY_DB_PATH", str(tmp_path / "buddy.db"))
    ctx = {
        "user": {"id": "u1", "account_uuid": "account-u1"},
        "organization": {"id": "o1", "organization_uuid": "org-u1"},
    }
    companion = get_companion(ctx, create=True)
    runtime = AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn", usage={})]),
        tools=ToolRegistry([]),
        context_builder=ContextBuilder(base_instructions="test"),
        config=AgentRuntimeConfig(
            session_id="s1",
            user_id="u1",
            org_id="o1",
            metadata={"account_uuid": "account-u1", "organization_uuid": "org-u1"},
        ),
    )

    events = [event async for event in runtime.run(f"hi {companion.name}")]

    reactions = [event for event in events if event.type == "companion_reaction"]
    assert reactions
    assert reactions[0].data["companion"]["name"] == companion.name
