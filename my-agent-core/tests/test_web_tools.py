from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.permissions.policy import StaticPermissionPolicy, get_session_permission_state, reset_session_permission_rules
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolContext, ToolRegistry
from agent_core.tools.web import WebFetchTool, WebSearchTool, _search_searxng
from agent_server.searxng_service import SearxngService
from agent_server.slash_commands import handle_slash_command


def test_web_tools_are_visible_without_deferred_activation() -> None:
    registry = ToolRegistry([WebFetchTool(), WebSearchTool()])

    names = {schema["name"] for schema in registry.schemas()}
    deferred_names = {schema["name"] for schema in registry.deferred_schemas()}

    assert "WebFetch" in names
    assert "WebSearch" in names
    assert "WebFetch" not in deferred_names
    assert "WebSearch" not in deferred_names


@pytest.mark.asyncio
async def test_web_fetch_returns_fetched_text_and_metadata(tmp_path: Path) -> None:
    async def fetcher(url: str) -> dict:
        return {
            "url": url,
            "code": 200,
            "codeText": "OK",
            "contentType": "text/html; charset=utf-8",
            "body": "<html><body><h1>Release Notes</h1><p>Version 2.0 shipped today.</p></body></html>",
        }

    result = await WebFetchTool(fetcher=fetcher).call(
        {"url": "https://example.com/releases", "prompt": "Summarize the release"},
        ToolContext(session_id="web-fetch-test", messages=[], cwd=str(tmp_path)),
    )

    assert result.is_error is False
    assert "Release Notes" in result.content
    assert "Version 2.0 shipped today." in result.content
    assert result.metadata["code"] == 200
    assert result.metadata["contentType"].startswith("text/html")


@pytest.mark.asyncio
async def test_web_fetch_reports_cross_host_redirect(tmp_path: Path) -> None:
    async def fetcher(_url: str) -> dict:
        return {
            "type": "redirect",
            "originalUrl": "https://example.com/start",
            "redirectUrl": "https://docs.example.org/final",
            "statusCode": 302,
        }

    result = await WebFetchTool(fetcher=fetcher).call(
        {"url": "https://example.com/start", "prompt": "Read it"},
        ToolContext(session_id="web-fetch-redirect-test", messages=[], cwd=str(tmp_path)),
    )

    assert result.is_error is False
    assert "REDIRECT DETECTED" in result.content
    assert result.metadata["redirectUrl"] == "https://docs.example.org/final"


@pytest.mark.asyncio
async def test_web_fetch_quotes_unicode_query_before_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200
        reason = "OK"
        headers = {"content-type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit: int) -> bytes:
            return "<html><body>微软 AI PC 新闻</body></html>".encode("utf-8")

        def geturl(self) -> str:
            return "https://www.baidu.com/s?wd=%E5%BE%AE%E8%BD%AF+AI+PC+%E6%96%B0%E9%97%BB"

    class FakeOpener:
        def open(self, req, timeout):
            assert req.full_url == "https://www.baidu.com/s?wd=%E5%BE%AE%E8%BD%AF+AI+PC+%E6%96%B0%E9%97%BB"
            req.full_url.encode("ascii")
            return FakeResponse()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: FakeOpener())

    result = await WebFetchTool().call(
        {"url": "https://www.baidu.com/s?wd=微软+AI+PC+新闻", "prompt": "提取新闻"},
        ToolContext(session_id="web-fetch-unicode-query-test", messages=[], cwd=str(tmp_path)),
    )

    assert result.is_error is False
    assert "微软 AI PC 新闻" in result.content


@pytest.mark.asyncio
async def test_web_search_filters_results_and_returns_sources(tmp_path: Path) -> None:
    async def searcher(query: str, allowed_domains: list[str] | None, blocked_domains: list[str] | None) -> list[dict[str, str]]:
        return [
            {"title": "Python docs", "url": "https://docs.python.org/3/library/asyncio.html", "snippet": "asyncio docs"},
            {"title": "Blocked", "url": "https://spam.example/result", "snippet": "spam"},
        ]

    result = await WebSearchTool(searcher=searcher).call(
        {"query": "python asyncio latest", "allowed_domains": ["docs.python.org"], "blocked_domains": []},
        ToolContext(session_id="web-search-test", messages=[], cwd=str(tmp_path)),
    )

    assert result.is_error is False
    assert "Python docs" in result.content
    assert "spam.example" not in result.content
    assert result.metadata["results"] == [
        {"title": "Python docs", "url": "https://docs.python.org/3/library/asyncio.html", "snippet": "asyncio docs"}
    ]


@pytest.mark.asyncio
async def test_web_tools_require_permission_and_accept_session_rules(tmp_path: Path) -> None:
    session_id = "web-permission-test"
    reset_session_permission_rules(session_id, reset_mode=True)
    policy = StaticPermissionPolicy(session_id=session_id, cwd=tmp_path)
    fetch_tool = WebFetchTool(fetcher=lambda _url: None)  # type: ignore[arg-type]
    search_tool = WebSearchTool(searcher=lambda _q, _a, _b: None)  # type: ignore[arg-type]

    fetch_decision = await policy.check(tool=fetch_tool, tool_input={"url": "https://example.com/a", "prompt": "read"})
    assert fetch_decision.status == "ask"
    assert fetch_decision.metadata["domain"] == "example.com"
    fetch_option = next(option for option in fetch_decision.options if option.get("type") == "accept-session")
    policy.record_user_decision(tool=fetch_tool, tool_input={"url": "https://example.com/a", "prompt": "read"}, option=fetch_option)
    assert get_session_permission_state(session_id).web_domains == {"example.com"}
    fetch_allowed = await policy.check(tool=fetch_tool, tool_input={"url": "https://www.example.com/b", "prompt": "read"})
    assert fetch_allowed.status == "allow"

    search_decision = await policy.check(tool=search_tool, tool_input={"query": "latest python"})
    assert search_decision.status == "ask"
    search_option = next(option for option in search_decision.options if option.get("type") == "accept-session")
    policy.record_user_decision(tool=search_tool, tool_input={"query": "latest python"}, option=search_option)
    assert get_session_permission_state(session_id).web_search_allowed is True
    search_allowed = await policy.check(tool=search_tool, tool_input={"query": "latest python"})
    assert search_allowed.status == "allow"


@pytest.mark.asyncio
async def test_permissions_command_manages_web_rules(tmp_path: Path) -> None:
    session_id = "web-permissions-command-test"
    store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")

    async def not_running(_session_id: str) -> bool:
        return False

    async def abort(_session_id: str) -> bool:
        return True

    web_result = await handle_slash_command(
        "/permissions allow-web https://www.example.com/docs",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    search_result = await handle_slash_command(
        "/permissions allow-web-search",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )

    assert web_result["permission_rules"]["web_domains"] == ["example.com"]
    assert search_result["permission_rules"]["web_search_allowed"] is True
    saved = store.get_session(session_id)["metadata"]["permissions"]
    assert saved["web_domains"] == ["example.com"]
    assert saved["web_search_allowed"] is True

    revoke_result = await handle_slash_command(
        "/permissions revoke-web example.com",
        session_id=session_id,
        session_store=store,
        is_running=not_running,
        abort=abort,
    )
    assert revoke_result["permission_rules"]["web_domains"] == []


def test_searxng_service_writes_json_enabled_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "vendor" / "searxng"
    source.mkdir(parents=True)
    monkeypatch.setenv("AGENT_SEARXNG_SOURCE_DIR", str(source))
    monkeypatch.setenv("AGENT_SEARXNG_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_SEARXNG_PORT", "19999")
    service = SearxngService(project_root=tmp_path)

    service._write_settings()  # runtime-generated config is intentionally tested as a stable contract

    settings = (tmp_path / "state" / "settings.yml").read_text(encoding="utf-8")
    assert "use_default_settings: true" in settings
    assert "    - json" in settings
    assert "port: 19999" in settings
    assert "limiter: false" in settings


def test_search_searxng_maps_json_results(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _limit: int) -> bytes:
            return (
                '{"results":[{"title":"Result A","url":"https://example.com/a","content":"Snippet A","engine":"duckduckgo"}]}'
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        assert "format=json" in req.full_url
        assert "/search?" in req.full_url
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    results = _search_searxng("http://127.0.0.1:18888", "hello", None, None)

    assert results == [
        {"title": "Result A", "url": "https://example.com/a", "snippet": "Snippet A", "engine": "duckduckgo"}
    ]
