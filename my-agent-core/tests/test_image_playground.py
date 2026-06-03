from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_core.attachments import ImageAttachmentStore
from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.images import GeneratedImage, ImageGenerationResult, ImageInput
from agent_core.images.openai_compatible import CODEX_PROMPT_PREFIX, OpenAICompatibleImageGenerationClient
from agent_core.images.top_aidp import AIDPImageGenerationClient, TopAidpImageGenerationClient, build_top_common_params, sign_top_request, sign_volcengine_request
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.image_generation import GenerateImageTool
from agent_core.types import TextBlock, ToolUseBlock


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"1" * 32


class FakeImageClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_image(self, **kwargs: Any) -> ImageGenerationResult:
        self.calls.append({"op": "generate", **kwargs})
        return ImageGenerationResult(
            images=[GeneratedImage(data=PNG_BYTES, content_type="image/png", revised_prompt="revised")],
            metadata={"model": "fake-image", "requested": kwargs},
        )

    async def edit_image(self, **kwargs: Any) -> ImageGenerationResult:
        self.calls.append({"op": "edit", **kwargs})
        return ImageGenerationResult(
            images=[GeneratedImage(data=PNG_BYTES, content_type="image/png", revised_prompt="edited")],
            metadata={"model": "fake-image", "requested": {k: v for k, v in kwargs.items() if k != "input_images"}},
        )


class ConcurrentFakeImageClient(FakeImageClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def generate_image(self, **kwargs: Any) -> ImageGenerationResult:
        self.calls.append({"op": "generate", **kwargs})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            prompt = str(kwargs.get("prompt") or "image")
            await asyncio.sleep(0.02 if prompt == "image-a" else 0.05)
            return ImageGenerationResult(
                images=[GeneratedImage(data=PNG_BYTES + prompt.encode(), content_type="image/png", revised_prompt=prompt)],
                metadata={"model": "fake-image"},
            )
        finally:
            self.active -= 1

@pytest.mark.asyncio
async def test_codex_client_prefixes_prompt_and_omits_quality_async(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, **kwargs: Any):
            captured["url"] = url
            captured["kwargs"] = kwargs
            payload = {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode("ascii"), "revised_prompt": "x"}]}
            return types.SimpleNamespace(status_code=200, text="{}", json=lambda: payload)

        async def get(self, url: str):  # pragma: no cover - not used in this test
            raise AssertionError("unexpected get")

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = OpenAICompatibleImageGenerationClient(
        model="gpt-image-2",
        api_key="key",
        base_url="https://example.test/v1",
        codex_cli=True,
    )
    result = await client.generate_image(prompt="draw a fox", quality="high")

    assert result.images[0].data == PNG_BYTES
    body = captured["kwargs"]["json"]
    assert captured["url"] == "https://example.test/v1/images/generations"
    assert body["prompt"].startswith(CODEX_PROMPT_PREFIX)
    assert "quality" not in body
    assert body["response_format"] == "b64_json"


def test_image_playground_generate_endpoint_saves_attachment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_server.app as app_module
    import agent_server.runtime_factory as runtime_factory

    session_store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    attachment_store = ImageAttachmentStore(tmp_path / "attachments")
    fake_client = FakeImageClient()
    session_store.create_session(session_id="s1", user_id="local-user", org_id="local-org", metadata={"user_id": "local-user", "org_id": "local-org"})

    monkeypatch.setattr(app_module, "create_session_store", lambda: session_store)
    monkeypatch.setattr(app_module, "get_attachment_store", lambda: attachment_store)
    monkeypatch.setattr(app_module, "get_image_generation_client", lambda: fake_client)
    monkeypatch.setattr(runtime_factory, "_attachment_store", attachment_store)

    client = TestClient(app_module.app)
    response = client.post(
        "/terminal/api/sessions/s1/image-playground/generate",
        headers={"x-terminal-token": app_module.get_terminal_token(), "x-agent-user-id": "local-user", "x-agent-org-id": "local-org"},
        json={"prompt": "draw a fox", "size": "1024x1024", "n": 1},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    attachment = payload["attachments"][0]
    assert attachment["id"].startswith("img_")
    assert attachment["preview_url"].endswith(f"/attachments/{attachment['id']}/content")
    saved = attachment_store.get(attachment["id"])
    assert saved is not None
    assert saved.metadata["source"] == "image-playground"
    assert saved.metadata["operation"] == "generate"
    assert attachment_store.read_bytes(saved) == PNG_BYTES


def test_image_playground_edit_requires_authorized_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_server.app as app_module

    session_store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    attachment_store = ImageAttachmentStore(tmp_path / "attachments")
    fake_client = FakeImageClient()
    session_store.create_session(session_id="s1", user_id="local-user", org_id="local-org", metadata={"user_id": "local-user", "org_id": "local-org"})
    other = attachment_store.save_image(data=PNG_BYTES, filename="other.png", content_type="image/png", user_id="local-user", org_id="local-org", session_id="s2")

    monkeypatch.setattr(app_module, "create_session_store", lambda: session_store)
    monkeypatch.setattr(app_module, "get_attachment_store", lambda: attachment_store)
    monkeypatch.setattr(app_module, "get_image_generation_client", lambda: fake_client)

    client = TestClient(app_module.app)
    response = client.post(
        "/terminal/api/sessions/s1/image-playground/edit",
        headers={"x-terminal-token": app_module.get_terminal_token(), "x-agent-user-id": "local-user", "x-agent-org-id": "local-org"},
        json={"prompt": "edit", "mode": "edit", "input_attachment_ids": [other.id]},
    )

    assert response.status_code == 404
    assert "input attachment not found" in response.text


@pytest.mark.asyncio
async def test_runtime_runs_multiple_generate_image_tools_in_parallel(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    fake = ConcurrentFakeImageClient()
    tool = GenerateImageTool(store, lambda: fake)
    runtime = AgentRuntime(
        model=ScriptedModelClient(
            [
                ModelResponse(
                    content=[
                        ToolUseBlock(id="tool-a", name="GenerateImage", input={"prompt": "image-a"}),
                        ToolUseBlock(id="tool-b", name="GenerateImage", input={"prompt": "image-b"}),
                    ],
                    stop_reason="tool_use",
                ),
                ModelResponse(content=[TextBlock("done")], stop_reason="end_turn"),
            ]
        ),
        tools=ToolRegistry([tool]),
        context_builder=ContextBuilder(base_instructions="test"),
        config=AgentRuntimeConfig(session_id="session-1", user_id="user-1", org_id="org-1", cwd=str(tmp_path)),
    )

    events = [event async for event in runtime.run("同时生成两张图")]

    assert fake.max_active == 2
    starts = [event for event in events if event.type == "tool_started"]
    completes = [event for event in events if event.type == "tool_completed"]
    assert [event.data["tool_use_id"] for event in starts] == ["tool-a", "tool-b"]
    assert all(event.data["concurrent"] is True for event in starts + completes)
    assert [event.data["tool_use_id"] for event in completes] == ["tool-a", "tool-b"]
    assert all(event.data["metadata"]["attachments"] for event in completes)


def test_image_provider_uses_explicit_agent_image_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_server.runtime_factory as runtime_factory

    captured: dict[str, Any] = {}

    class FakeOpenAICompatibleImageGenerationClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("AGENT_IMAGE_MODEL", "gpt-image-explicit")
    monkeypatch.setenv("AGENT_IMAGE_BASE_URL", "http://image.local/v1")
    monkeypatch.setenv("AGENT_IMAGE_API_KEY", "image-key")
    monkeypatch.setenv("AGENT_IMAGE_PROVIDER", "image")
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_core.images.openai_compatible",
        types.SimpleNamespace(OpenAICompatibleImageGenerationClient=FakeOpenAICompatibleImageGenerationClient),
    )
    runtime_factory.clear_image_generation_client_cache()

    client = runtime_factory.get_image_generation_client()

    assert client is not None
    assert captured["model"] == "gpt-image-explicit"
    assert captured["base_url"] == "http://image.local/v1"
    assert captured["api_key"] == "image-key"
    runtime_factory.clear_image_generation_client_cache()


def test_top_aidp_sign_top_request_matches_java_style() -> None:
    params = build_top_common_params(app_key="app-key", method="CreateImageTask", timestamp="2026-06-01 12:00:00")
    body = json.dumps({"prompt": "fox"}, ensure_ascii=False, separators=(",", ":"))

    source = "".join(f"{key}{params[key]}" for key in sorted(params)) + body
    expected = hmac.new("secret".encode("utf-8"), source.encode("utf-8"), hashlib.sha256).hexdigest().upper()

    assert sign_top_request(params=params, body=body, app_secret="secret") == expected


def test_top_aidp_volsign_headers_are_stable() -> None:
    import datetime as dt

    headers = sign_volcengine_request(
        method="POST",
        path="/",
        query={"Action": "CreateImageTask", "Version": "2022-08-01"},
        body='{"prompt":"fox"}',
        host="open.volcengineapi.com",
        access_key_id="ak",
        secret_access_key="sk",
        service="cdp_saas",
        region="cn-beijing",
        now=dt.datetime(2026, 6, 1, 12, 0, 0),
    )

    assert headers["X-Date"] == "20260601T120000Z"
    assert headers["X-Content-Sha256"] == hashlib.sha256(b'{"prompt":"fox"}').hexdigest()
    assert headers["Authorization"].startswith("HMAC-SHA256 Credential=ak/20260601/cn-beijing/cdp_saas/request")
    assert "SignedHeaders=host;x-date;x-content-sha256;content-type" in headers["Authorization"]


def test_aidp_edit_payload_compresses_large_input_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pil_image = pytest.importorskip("PIL.Image")

    raw_pixels = bytes((idx * 37 + idx // 7) % 256 for idx in range(900 * 900 * 3))
    image = pil_image.frombytes("RGB", (900, 900), raw_pixels)
    raw = io.BytesIO()
    image.save(raw, format="BMP")
    raw_bytes = raw.getvalue()

    monkeypatch.setenv("AGENT_IMAGE_EDIT_MAX_INPUT_BYTES", "300000")
    monkeypatch.setenv("AGENT_IMAGE_EDIT_MAX_INPUT_DIMENSION", "256")
    client = AIDPImageGenerationClient(access_key_id="ak", secret_access_key="sk", token="token")

    payload = client._create_payload(
        prompt="改成蓝色背景",
        size="1024x1024",
        quality=None,
        n=1,
        input_images=[ImageInput(data=raw_bytes, content_type="image/bmp", filename="source.bmp")],
        metadata=None,
    )

    encoded = payload["images"][0]["b64"]
    decoded = base64.b64decode(encoded)
    assert len(raw_bytes) > 300000
    assert len(decoded) <= 300000
    assert payload["images"][0]["content_type"] == "image/jpeg"
    assert payload["images"][0]["filename"] == "source.jpg"


@pytest.mark.asyncio
async def test_top_aidp_client_creates_task_polls_and_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_posts: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, **kwargs: Any):
            captured_posts.append({"url": url, **kwargs})
            method = kwargs["params"]["method"]
            if method == "CreateImageTask":
                payload = {"data": {"task_id": "task-1"}}
            else:
                payload = {"data": {"status": "done", "image_urls": ["https://image.test/out.png"]}}
            return types.SimpleNamespace(status_code=200, text="{}", json=lambda: payload)

        async def get(self, url: str):
            assert url == "https://image.test/out.png"
            return types.SimpleNamespace(status_code=200, content=PNG_BYTES, headers={"content-type": "image/png"})

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = TopAidpImageGenerationClient(
        app_key="app-key",
        app_secret="secret",
        base_url="https://top.test/router/rest",
        model="aidp-model",
        poll_interval=0,
    )
    result = await client.generate_image(prompt="draw a fox", size="1536x1024", n=2)

    assert result.images[0].data == PNG_BYTES
    create_call = captured_posts[0]
    assert create_call["url"] == "https://top.test/router/rest"
    assert create_call["params"]["method"] == "CreateImageTask"
    assert create_call["params"]["app_key"] == "app-key"
    assert create_call["params"]["sign_method"] == "hmac-sha256"
    assert create_call["params"]["sign"]
    body = json.loads(create_call["content"].decode("utf-8"))
    assert body["prompt"] == "draw a fox"
    assert body["width"] == 1536
    assert body["height"] == 1024
    assert body["n"] == 2
    assert body["model"] == "aidp-model"


@pytest.mark.asyncio
async def test_top_aidp_client_uses_volsign_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_posts: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, **kwargs: Any):
            captured_posts.append({"url": url, **kwargs})
            action = kwargs["params"]["Action"]
            if action == "CreateImageTask":
                payload = {"Result": {"task_id": "task-1"}}
            else:
                payload = {"Result": {"status": "succeeded", "image_urls": ["https://image.test/out.png"]}}
            return types.SimpleNamespace(status_code=200, text="{}", json=lambda: payload)

        async def get(self, url: str):
            return types.SimpleNamespace(status_code=200, content=PNG_BYTES, headers={"content-type": "image/png"})

    monkeypatch.setitem(__import__("sys").modules, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    client = TopAidpImageGenerationClient(
        app_key="ak",
        app_secret="sk",
        base_url="https://open.volcengineapi.com",
        model="aidp-model",
        api_version="2022-08-01",
        signature_protocol="volsign",
        volc_service="cdp_saas",
        volc_region="cn-beijing",
        aidp_token="aidp-token",
        poll_interval=0,
    )

    result = await client.generate_image(prompt="draw a fox")

    assert result.images[0].data == PNG_BYTES
    create_call = captured_posts[0]
    assert create_call["url"] == "https://open.volcengineapi.com/"
    assert create_call["params"] == {"Action": "CreateImageTask", "Version": "2022-08-01"}
    assert create_call["headers"]["Authorization"].startswith("HMAC-SHA256 Credential=ak/")
    assert "/cn-beijing/cdp_saas/request" in create_call["headers"]["Authorization"]
    assert create_call["headers"]["X-Content-Sha256"]
    create_body = json.loads(create_call["content"].decode("utf-8"))
    assert create_body["token"] == "aidp-token"
    status_body = json.loads(captured_posts[1]["content"].decode("utf-8"))
    assert status_body["token"] == "aidp-token"


def test_image_provider_uses_top_aidp_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_server.runtime_factory as runtime_factory

    captured: dict[str, Any] = {}

    class FakeTopAidpImageGenerationClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("AGENT_IMAGE_PROVIDER", "top_aidp")
    monkeypatch.setenv("TOP_AIDP_BASE_URL", "https://top.test/router/rest")
    monkeypatch.setenv("TOP_AIDP_APP_KEY", "app-key")
    monkeypatch.setenv("TOP_AIDP_APP_SECRET", "secret")
    monkeypatch.setenv("TOP_AIDP_MODEL", "aidp-model")
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_core.images.top_aidp",
        types.SimpleNamespace(TopAidpImageGenerationClient=FakeTopAidpImageGenerationClient),
    )
    runtime_factory.clear_image_generation_client_cache()

    client = runtime_factory.get_image_generation_client()

    assert client is not None
    assert captured["base_url"] == "https://top.test/router/rest"
    assert captured["app_key"] == "app-key"
    assert captured["app_secret"] == "secret"
    assert captured["model"] == "aidp-model"
    assert "aidp_token" in captured
    runtime_factory.clear_image_generation_client_cache()
