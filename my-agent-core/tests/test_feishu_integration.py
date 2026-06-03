from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_core.core.events import AgentEvent
from agent_core.integrations.feishu import FeishuApiTool, FeishuTokenStore
from agent_core.integrations.feishu_ws_bridge import FeishuWebSocketBridge, _BridgeHandle, _extract_inbound_message, _session_id_for_message
from agent_core.tools.base import ToolContext


def test_feishu_token_store_is_scoped_by_user_and_org(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")

    store.save_user_token(user_id="alice", org_id="org", access_token="alice-token", scopes=["im:message"])
    store.save_user_token(user_id="bob", org_id="org", access_token="bob-token", scopes=["doc:read"])

    alice_public = store.get_user_token(user_id="alice", org_id="org")
    bob_secret = store.get_user_token(user_id="bob", org_id="org", include_secret=True)

    assert alice_public is not None
    assert alice_public["user_id"] == "alice"
    assert alice_public["scopes"] == ["im:message"]
    assert "access_token" not in alice_public
    assert bob_secret is not None
    assert bob_secret["access_token"] == "bob-token"


def test_feishu_token_store_reports_expired_status(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_user_token(
        user_id="alice",
        org_id="org",
        access_token="alice-token",
        expires_at=time.time() - 1,
    )

    status = store.status(user_id="alice", org_id="org").to_dict()

    assert status["connected"] is False
    assert status["expires_at_iso"]


def test_feishu_token_store_saves_app_credentials_without_exposing_secret(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")

    store.save_app_credentials(user_id="alice", org_id="org", app_id="cli_xxx", app_secret="secret")

    public = store.get_user_token(user_id="alice", org_id="org")
    secret = store.get_user_token(user_id="alice", org_id="org", include_secret=True)
    status = store.status(user_id="alice", org_id="org").to_dict()

    assert public is not None
    assert public["credential_type"] == "app_credentials"
    assert public["app_id"] == "cli_xxx"
    assert "app_secret" not in public
    assert secret is not None
    assert secret["app_secret"] == "secret"
    assert status["connected"] is True
    assert status["credential_type"] == "app_credentials"


def test_feishu_token_store_updates_metadata_and_lists_tokens(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_app_credentials(user_id="alice", org_id="org", app_id="cli_xxx", app_secret="secret", metadata={"source": "test"})

    updated = store.update_metadata(user_id="alice", org_id="org", metadata={"bridge_autostart": True})
    tokens = store.list_tokens()

    assert updated is not None
    assert updated["metadata"]["source"] == "test"
    assert updated["metadata"]["bridge_autostart"] is True
    assert tokens[0]["user_id"] == "alice"
    assert tokens[0]["metadata"]["bridge_autostart"] is True
    assert "app_secret" not in tokens[0]


async def test_feishu_api_tool_uses_current_user_token_and_redacts_response(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_user_token(user_id="alice", org_id="org", access_token="alice-token")
    calls: list[dict[str, Any]] = []

    async def requester(method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        return {"code": 0, "data": {"name": "Alice", "access_token": "server-token"}}

    tool = FeishuApiTool(store, base_url="https://open.feishu.test", requester=requester)
    result = await tool.call(
        {"method": "GET", "path": "/open-apis/authen/v1/user_info", "query": {"lang": "zh"}},
        ToolContext(session_id="s1", messages=[], metadata={"user_id": "alice", "org_id": "org"}),
    )

    assert result.is_error is False
    assert calls[0]["headers"]["Authorization"] == "Bearer alice-token"
    assert calls[0]["url"] == "https://open.feishu.test/open-apis/authen/v1/user_info?lang=zh"
    payload = json.loads(result.content)
    assert payload["data"]["name"] == "Alice"
    assert payload["data"]["access_token"] == "[redacted]"


async def test_feishu_api_tool_exchanges_app_credentials_for_tenant_token(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_app_credentials(user_id="alice", org_id="org", app_id="cli_xxx", app_secret="secret")
    calls: list[dict[str, Any]] = []

    async def requester(method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
        calls.append({"method": method, "url": url, "headers": headers, "body": body})
        if url.endswith("/open-apis/auth/v3/tenant_access_token/internal"):
            assert json.loads((body or b"{}").decode("utf-8")) == {
                "app_id": "cli_xxx",
                "app_secret": "secret",
            }
            return {"code": 0, "tenant_access_token": "tenant-token"}
        return {"code": 0, "data": {"app_name": "My Bot"}}

    tool = FeishuApiTool(store, base_url="https://open.feishu.test", requester=requester)
    result = await tool.call(
        {"method": "GET", "path": "/open-apis/bot/v3/info"},
        ToolContext(session_id="s1", messages=[], metadata={"user_id": "alice", "org_id": "org"}),
    )

    assert result.is_error is False
    assert calls[1]["headers"]["Authorization"] == "Bearer tenant-token"
    payload = json.loads(result.content)
    assert payload["data"]["app_name"] == "My Bot"


async def test_feishu_api_tool_requires_connected_current_user(tmp_path: Path) -> None:
    tool = FeishuApiTool(FeishuTokenStore(tmp_path / "feishu.db"))

    result = await tool.call(
        {"method": "GET", "path": "/open-apis/authen/v1/user_info"},
        ToolContext(session_id="s1", messages=[], metadata={"user_id": "missing", "org_id": "org"}),
    )

    assert result.is_error is True
    assert "has not connected Feishu" in result.content


async def test_feishu_api_tool_rejects_non_openapi_paths(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_user_token(user_id="alice", org_id="org", access_token="alice-token")
    tool = FeishuApiTool(store)

    result = await tool.call(
        {"method": "GET", "path": "/evil"},
        ToolContext(session_id="s1", messages=[], metadata={"user_id": "alice", "org_id": "org"}),
    )

    assert result.is_error is True
    assert "path must start with /open-apis/" in result.content


def test_feishu_ws_bridge_extracts_text_message() -> None:
    inbound = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user", "user_id": "u_user"}},
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_1",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
            }
        }
    )

    assert inbound is not None
    assert inbound.message_id == "om_1"
    assert inbound.chat_id == "oc_1"
    assert inbound.text == "hello"
    assert inbound.sender_open_id == "ou_user"


async def test_feishu_ws_bridge_runs_agent_and_replies() -> None:
    prompts: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    class Runtime:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def run(self, prompt: str):
            prompts.append({"prompt": prompt, "kwargs": self.kwargs})
            yield AgentEvent("assistant_text", {"text": "agent reply"})
            yield AgentEvent("loop_completed", {})

    class ApiTool:
        async def call(self, tool_input: dict[str, Any], context: ToolContext):
            calls.append({"input": tool_input, "context": context})
            return type("Result", (), {"is_error": False, "content": json.dumps({"status": 200, "code": 0, "data": {"message_id": "reply_1"}})})()

    handle = _BridgeHandle(
        app_id="cli_xxx",
        app_secret="secret",
        user_id="alice",
        org_id="org",
        domain="https://open.feishu.test",
        runtime_factory=lambda **kwargs: Runtime(**kwargs),
        api_tool_factory=ApiTool,
    )

    await handle._handle_message_event(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user"}},
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_1",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "hi bot"}),
                },
            }
        }
    )

    assert prompts
    assert "hi bot" in prompts[0]["prompt"]
    assert prompts[0]["kwargs"]["user_id"] == "alice"
    assert prompts[0]["kwargs"]["org_id"] == "org"
    assert prompts[0]["kwargs"]["session_id"] == "feishu-org-alice-direct-ou_user"
    assert calls[0]["input"]["path"] == "/open-apis/im/v1/messages/om_1/reply"
    assert calls[0]["input"]["body"]["content"] == json.dumps({"text": "agent reply"}, ensure_ascii=False)
    assert handle.status.reply_count == 1
    assert handle.status.message_count == 1
    log_events = [item["event"] for item in handle.get_logs(limit=50)]
    assert "message_received" in log_events
    assert "agent_run_success" in log_events
    assert "reply_primary_success" in log_events


async def test_feishu_ws_bridge_ignores_app_sender() -> None:
    runtime_calls = 0

    handle = _BridgeHandle(
        app_id="cli_xxx",
        app_secret="secret",
        user_id="alice",
        org_id="org",
        domain="https://open.feishu.test",
        runtime_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("runtime should not run")),
        api_tool_factory=lambda: (_ for _ in ()).throw(AssertionError("api should not run")),
    )

    await handle._handle_message_event(
        {
            "event": {
                "sender": {"sender_type": "app", "sender_id": {"open_id": "ou_bot"}},
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_1",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "bot echo"}),
                },
            }
        }
    )

    assert runtime_calls == 0
    assert handle.status.ignored_count == 1
    assert handle.status.last_ignored_reason == "sender_is_app"
    assert handle.status.reply_count == 0


async def test_feishu_ws_bridge_releases_dedupe_when_agent_fails() -> None:
    calls: list[dict[str, Any]] = []
    runtime_runs = 0

    class Runtime:
        async def run(self, prompt: str):
            nonlocal runtime_runs
            runtime_runs += 1
            yield AgentEvent("loop_failed", {"error": "model down"})

    class ApiTool:
        async def call(self, tool_input: dict[str, Any], context: ToolContext):
            calls.append({"input": tool_input, "context": context})
            return type("Result", (), {"is_error": False, "content": json.dumps({"status": 200, "code": 0})})()

    handle = _BridgeHandle(
        app_id="cli_xxx",
        app_secret="secret",
        user_id="alice",
        org_id="org",
        domain="https://open.feishu.test",
        runtime_factory=lambda **kwargs: Runtime(),
        api_tool_factory=ApiTool,
    )
    event = {
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user"}},
            "message": {
                "message_id": "om_fail",
                "chat_id": "oc_1",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "hi bot"}),
            },
        }
    }

    await handle._handle_message_event(event)
    await handle._handle_message_event(event)

    assert runtime_runs == 2
    assert len(calls) == 2
    assert "处理这条消息时出错了" in json.loads(calls[0]["input"]["body"]["content"])["text"]
    assert handle.status.last_error == "model down"


async def test_feishu_ws_bridge_prefetches_bot_identity_from_bot_payload() -> None:
    class ApiTool:
        async def call(self, tool_input: dict[str, Any], context: ToolContext):
            return type(
                "Result",
                (),
                {
                    "is_error": False,
                    "content": json.dumps(
                        {"status": 200, "code": 0, "msg": "ok", "bot": {"open_id": "ou_bot", "app_name": "洞察助手dev"}}
                    ),
                },
            )()

    handle = _BridgeHandle(
        app_id="cli_xxx",
        app_secret="secret",
        user_id="alice",
        org_id="org",
        domain="https://open.feishu.test",
        runtime_factory=lambda **kwargs: None,  # type: ignore[arg-type]
        api_tool_factory=ApiTool,
    )

    await handle._prefetch_bot_identity()

    assert handle.status.bot_open_id == "ou_bot"
    assert handle.status.bot_name == "洞察助手dev"


def test_feishu_ws_bridge_requires_app_credentials(tmp_path: Path) -> None:
    store = FeishuTokenStore(tmp_path / "feishu.db")
    store.save_user_token(user_id="alice", org_id="org", access_token="user-token")
    bridge = FeishuWebSocketBridge(token_store=store, runtime_factory=lambda **kwargs: None)  # type: ignore[arg-type]

    try:
        bridge.start(user_id="alice", org_id="org")
    except ValueError as exc:
        assert "App ID + App Secret" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_feishu_ws_bridge_logs_returns_empty_without_running_handle(tmp_path: Path) -> None:
    bridge = FeishuWebSocketBridge(token_store=FeishuTokenStore(tmp_path / "feishu.db"), runtime_factory=lambda **kwargs: None)  # type: ignore[arg-type]

    payload = bridge.logs(user_id="alice", org_id="org", limit=10)

    assert payload == {"user_id": "alice", "org_id": "org", "logs": []}


def test_feishu_ws_bridge_session_id_is_stable_per_conversation() -> None:
    first = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user"}},
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_dm_chat_a",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "第一句"}),
                },
            }
        }
    )
    second = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user"}},
                "message": {
                    "message_id": "om_2",
                    "chat_id": "oc_dm_chat_b",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "第二句"}),
                },
            }
        }
    )
    other_sender = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_other"}},
                "message": {
                    "message_id": "om_3",
                    "chat_id": "oc_dm_chat_c",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "别人"}),
                },
            }
        }
    )

    assert first is not None and second is not None and other_sender is not None
    assert _session_id_for_message(first, user_id="alice", org_id="org") == _session_id_for_message(second, user_id="alice", org_id="org")
    assert _session_id_for_message(first, user_id="alice", org_id="org") != _session_id_for_message(other_sender, user_id="alice", org_id="org")


def test_feishu_ws_bridge_session_id_is_stable_per_group_chat() -> None:
    first = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user_1"}},
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_group",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": json.dumps({"text": "第一句"}),
                },
            }
        }
    )
    second = _extract_inbound_message(
        {
            "event": {
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_user_2"}},
                "message": {
                    "message_id": "om_2",
                    "chat_id": "oc_group",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": json.dumps({"text": "第二句"}),
                },
            }
        }
    )

    assert first is not None and second is not None
    assert _session_id_for_message(first, user_id="alice", org_id="org") == "feishu-org-alice-group-oc_group"
    assert _session_id_for_message(first, user_id="alice", org_id="org") == _session_id_for_message(second, user_id="alice", org_id="org")
