from __future__ import annotations

import json
import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agent_core.attachments import ImageAttachmentStore
from agent_core.adapters.web import event_to_chat_json
from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.core.events import AgentEvent
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.session.store import deserialize_message, serialize_message
from agent_core.tools.base import ToolContext, ToolRegistry, ToolResult
from agent_core.tools.video_api import VideoApiTool
from agent_core.tools.video_generation import GenerateVideoTool
from agent_core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from agent_core.videos import GeneratedVideo, VideoGenerationResult
from agent_core.videos.rest_bearer import BearerVideoGenerationClient


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"1" * 32
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32
MP3_BYTES = b"ID3" + b"2" * 32


@pytest.mark.asyncio
async def test_bearer_video_client_creates_polls_and_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any):
            calls.append({"method": method, "url": url, **kwargs})
            if method == "POST" and url.endswith("/create"):
                payload = {"task_id": "cgt-1", "status": "running", "mode": "fast"}
            else:
                payload = {"task_id": "cgt-1", "status": "succeeded", "video_url": "https://video.test/out.mp4", "duration": 5}
            return types.SimpleNamespace(status_code=200, text=json.dumps(payload), json=lambda: payload)

        async def get(self, url: str):
            assert url == "https://video.test/out.mp4"
            return types.SimpleNamespace(status_code=200, content=MP4_BYTES, headers={"content-type": "video/mp4"})

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    client = BearerVideoGenerationClient(base_url="https://api.video.test", token="token", poll_interval=0)

    result = await client.generate_video(prompt="星空下的沙漠", mode="fast", ratio="16:9", duration=5)

    assert result.videos[0].data == MP4_BYTES
    assert result.videos[0].task_id == "cgt-1"
    create = calls[0]
    assert create["url"] == "https://api.video.test/create"
    assert create["headers"]["Authorization"] == "Bearer token"
    assert create["json"]["prompt"] == "星空下的沙漠"
    assert create["json"]["ratio"] == "16:9"


@pytest.mark.asyncio
async def test_bearer_video_client_downloads_multiple_result_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any):
            if method == "POST" and url.endswith("/create"):
                payload = {"task_id": "cgt-2", "status": "running"}
            else:
                payload = {"task_id": "cgt-2", "status": "succeeded", "video_urls": ["https://video.test/a.mp4", "https://video.test/b.mp4"]}
            return types.SimpleNamespace(status_code=200, text=json.dumps(payload), json=lambda: payload)

        async def get(self, url: str):
            return types.SimpleNamespace(status_code=200, content=url.encode(), headers={"content-type": "video/mp4"})

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    client = BearerVideoGenerationClient(base_url="https://api.video.test", token="token", poll_interval=0)

    result = await client.generate_video(prompt="生成多条视频", mode="fast")

    assert len(result.videos) == 2
    assert result.videos[0].raw_url == "https://video.test/a.mp4"
    assert result.videos[1].raw_url == "https://video.test/b.mp4"


class FakeVideoClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_video(self, **kwargs: Any) -> VideoGenerationResult:
        self.calls.append(kwargs)
        return VideoGenerationResult(
            videos=[GeneratedVideo(data=MP4_BYTES, content_type="video/mp4", raw_url="https://video.test/out.mp4", task_id="cgt-1")],
            metadata={"task_id": "cgt-1"},
        )


class FailingVideoClient:
    async def generate_video(self, **kwargs: Any) -> VideoGenerationResult:
        raise RuntimeError("provider rejected request: bad reference url body={\"detail\":\"expired signed url\"}")


class SlowFakeVideoClient(FakeVideoClient):
    async def generate_video(self, **kwargs: Any) -> VideoGenerationResult:
        await asyncio.sleep(0.035)
        return await super().generate_video(**kwargs)


class ConcurrentFakeVideoClient(FakeVideoClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def generate_video(self, **kwargs: Any) -> VideoGenerationResult:
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            prompt = str(kwargs.get("prompt") or "video")
            await asyncio.sleep(0.02 if prompt == "video-a" else 0.06)
            return VideoGenerationResult(
                videos=[GeneratedVideo(data=MP4_BYTES + prompt.encode(), content_type="video/mp4", raw_url=f"https://video.test/{prompt}.mp4", task_id=prompt)],
                metadata={"task_id": prompt},
            )
        finally:
            self.active -= 1


class LimitedConcurrentTool:
    name = "LimitedConcurrent"
    description = "Test-only concurrency limited tool"
    input_schema = {"type": "object", "properties": {"delay": {"type": "number"}}}
    is_concurrency_safe = True
    concurrency_group = "limited-test"
    max_concurrency = 2
    should_defer = False

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(float(tool_input.get("delay") or 0.01))
            return ToolResult(content=f"ok {context.metadata.get('tool_use_id')}")
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_generate_video_tool_saves_video_attachment_with_image_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("TOS_AK", "TOS_SK", "TOS_BUCKET", "AGENT_TOS_AK", "AGENT_TOS_SK", "AGENT_TOS_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    store = ImageAttachmentStore(tmp_path)
    image = store.save_media(
        data=PNG_BYTES,
        filename="ref.png",
        content_type="image/png",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    fake = FakeVideoClient()
    tool = GenerateVideoTool(store, lambda: fake)

    result = await tool.call(
        {
            "prompt": "图片1中的角色转身微笑",
            "mode": "pro",
            "ratio": "9:16",
            "duration": 6,
            "image_attachment_ids": [image.id],
            "image_role": "first_frame",
            "pe_mode": "reference",
        },
        ToolContext(session_id="session-1", messages=[], metadata={"user_id": "user-1", "org_id": "org-1"}, cwd=str(tmp_path)),
    )

    assert result.is_error is False
    attachment = result.metadata["attachments"][0]
    assert attachment["id"].startswith("vid_")
    assert store.read_bytes(store.get(attachment["id"])) == MP4_BYTES  # type: ignore[arg-type]
    call = fake.calls[0]
    assert call["mode"] == "pro"
    assert call["ratio"] == "9:16"
    assert call["images"][0].role == "first_frame"
    assert call["images"][0].url.startswith("data:image/png;base64,")
    assert call["metadata"]["pe_mode"] == "reference"


class FakeTosStore:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.presigned: list[str] = []

    async def upload_bytes(self, **kwargs: Any):
        self.uploads.append(kwargs)
        kind = kwargs["media_kind"]
        return types.SimpleNamespace(
            key=f"org-1/user-1/session-1/assets/2026-06-01/{kind}_1.bin",
            stored_url=f"tos://bucket/{kind}_1.bin",
            signed_url=f"https://tos.test/{kind}_1.bin?sign=1",
            expires=86400,
        )

    async def presign_key(self, key: str):
        self.presigned.append(key)
        return f"https://tos.test/{key}?sign=1"

    def parse_tos_uri(self, value: str) -> str | None:
        assert value.startswith("tos://bucket/")
        return value.removeprefix("tos://bucket/")


@pytest.mark.asyncio
async def test_generate_video_tool_uploads_video_audio_and_image_attachments_to_tos(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    image = store.save_media(data=PNG_BYTES, filename="ref.png", content_type="image/png", user_id="user-1", org_id="org-1", session_id="session-1")
    video = store.save_media(data=MP4_BYTES, filename="ref.mp4", content_type="video/mp4", user_id="user-1", org_id="org-1", session_id="session-1")
    audio = store.save_media(data=MP3_BYTES, filename="voice.mp3", content_type="audio/mpeg", user_id="user-1", org_id="org-1", session_id="session-1")
    fake = FakeVideoClient()
    tos = FakeTosStore()
    tool = GenerateVideoTool(store, lambda: fake, tos_store=tos)  # type: ignore[arg-type]

    result = await tool.call(
        {
            "prompt": "用参考视频和旁白生成广告片",
            "mode": "pro",
            "image_attachment_ids": [image.id],
            "video_attachment_ids": [video.id],
            "audio_attachment_ids": [audio.id],
            "image_urls": ["tos://bucket/existing/ref.png"],
            "image_role": "first_frame",
        },
        ToolContext(session_id="session-1", messages=[], metadata={"user_id": "user-1", "org_id": "org-1"}, cwd=str(tmp_path)),
    )

    assert result.is_error is False
    call = fake.calls[0]
    assert call["images"][0].url == "https://tos.test/existing/ref.png?sign=1"
    assert call["images"][1].url == "https://tos.test/image_1.bin?sign=1"
    assert call["videos"][0].url == "https://tos.test/video_1.bin?sign=1"
    assert call["audios"][0].url == "https://tos.test/audio_1.bin?sign=1"
    assert call["videos"][0].role == "reference_video"
    assert call["audios"][0].role == "reference_audio"
    assert [upload["media_kind"] for upload in tos.uploads] == ["image", "video", "audio"]
    assert call["metadata"]["uploaded_tos_assets"][0]["stored_url"] == "tos://bucket/image_1.bin"


@pytest.mark.asyncio
async def test_generate_video_tool_returns_full_provider_exception_to_agent(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    tool = GenerateVideoTool(store, lambda: FailingVideoClient())

    result = await tool.call(
        {"prompt": "生成视频"},
        ToolContext(session_id="session-1", messages=[], metadata={"user_id": "user-1", "org_id": "org-1"}, cwd=str(tmp_path)),
    )

    assert result.is_error is True
    assert "GenerateVideo failed during provider request or result download" in result.content
    assert "RuntimeError: provider rejected request" in result.content
    assert "expired signed url" in result.content
    assert "Traceback:" in result.content
    assert result.metadata["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_generate_video_tool_emits_heartbeat_progress_for_long_generation(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    fake = SlowFakeVideoClient()
    tool = GenerateVideoTool(store, lambda: fake, heartbeat_interval=0.01)
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    result = await tool.call(
        {"prompt": "长视频生成心跳", "mode": "pro"},
        ToolContext(
            session_id="session-1",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1", "tool_use_id": "tool-1"},
            cwd=str(tmp_path),
            event_callback=emit,
        ),
    )

    assert result.is_error is False
    progress = [event for event in events if event.type == "tool_progress"]
    assert progress
    assert progress[0].data["tool"] == "GenerateVideo"
    assert progress[0].data["tool_use_id"] == "tool-1"


@pytest.mark.asyncio
async def test_runtime_runs_multiple_generate_video_tools_in_parallel_with_progress(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    fake = ConcurrentFakeVideoClient()
    tool = GenerateVideoTool(store, lambda: fake, heartbeat_interval=0.01)
    runtime = AgentRuntime(
        model=ScriptedModelClient(
            [
                ModelResponse(
                    content=[
                        ToolUseBlock(id="tool-a", name="GenerateVideo", input={"prompt": "video-a"}),
                        ToolUseBlock(id="tool-b", name="GenerateVideo", input={"prompt": "video-b"}),
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

    events = [event async for event in runtime.run("同时生成两条视频")]

    assert fake.max_active == 2
    starts = [event for event in events if event.type == "tool_started"]
    completes = [event for event in events if event.type == "tool_completed"]
    progress = [event for event in events if event.type == "tool_progress"]
    assert [event.data["tool_use_id"] for event in starts] == ["tool-a", "tool-b"]
    assert all(event.data["concurrent"] is True for event in starts + completes)
    assert {event.data["tool_use_id"] for event in progress} == {"tool-a", "tool-b"}
    assert [event.data["tool_use_id"] for event in completes] == ["tool-a", "tool-b"]
    event_keys = [(event.type, event.data.get("tool_use_id")) for event in events]
    assert event_keys.index(("tool_started", "tool-b")) < event_keys.index(("tool_completed", "tool-a"))
    assert all(event.data["metadata"]["attachments"] for event in completes)


@pytest.mark.asyncio
async def test_runtime_applies_generic_per_tool_concurrency_limit(tmp_path: Path) -> None:
    tool = LimitedConcurrentTool()
    runtime = AgentRuntime(
        model=ScriptedModelClient(
            [
                ModelResponse(
                    content=[
                        ToolUseBlock(id=f"tool-{idx}", name="LimitedConcurrent", input={"delay": 0.02})
                        for idx in range(5)
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

    events = [event async for event in runtime.run("run limited tools")]

    assert tool.max_active == 2
    assert len([event for event in events if event.type == "tool_started"]) == 5
    assert len([event for event in events if event.type == "tool_completed"]) == 5


def test_runtime_creates_video_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_server.runtime_factory as runtime_factory

    captured: dict[str, Any] = {}

    class FakeBearerVideoGenerationClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("AGENT_VIDEO_BASE_URL", "https://api.video.test")
    monkeypatch.setenv("AGENT_VIDEO_TOKEN", "token")
    monkeypatch.setitem(sys.modules, "agent_core.videos.rest_bearer", types.SimpleNamespace(BearerVideoGenerationClient=FakeBearerVideoGenerationClient))
    runtime_factory.clear_video_generation_client_cache()

    client = runtime_factory.get_video_generation_client()

    assert client is not None
    assert captured["base_url"] == "https://api.video.test"
    assert captured["token"] == "token"
    runtime_factory.clear_video_generation_client_cache()


def test_tool_result_block_persists_generated_video_attachment_metadata() -> None:
    message = Message(
        role="user",
        content=[
            ToolResultBlock(
                tool_use_id="tool-1",
                content="GenerateVideo completed. Saved video attachment IDs:\n- vid_1",
                metadata={"attachments": [{"id": "vid_1", "filename": "edited.mp4", "content_type": "video/mp4"}]},
            )
        ],
    )

    restored = deserialize_message(serialize_message(message))

    block = restored.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.metadata["attachments"][0]["id"] == "vid_1"
    assert block.metadata["attachments"][0]["content_type"] == "video/mp4"


def test_tool_completed_chat_event_keeps_video_attachment_metadata() -> None:
    payload = event_to_chat_json(
        AgentEvent(
            "tool_completed",
            {
                "tool_use_id": "tool-1",
                "tool": "GenerateVideo",
                "is_error": False,
                "result": "GenerateVideo completed",
                "metadata": {"attachments": [{"id": "vid_1", "filename": "out.mp4", "content_type": "video/mp4"}]},
            },
        )
    )

    assert payload is not None
    data = json.loads(payload)
    assert data["type"] == "tool_completed"
    assert data["metadata"]["attachments"][0]["id"] == "vid_1"
    assert data["metadata"]["attachments"][0]["content_type"] == "video/mp4"


def test_terminal_media_gallery_keeps_original_aspect_and_success_text() -> None:
    html = (Path(__file__).resolve().parents[1] / "src" / "agent_server" / "static" / "terminal.html").read_text(encoding="utf-8")

    assert "object-fit:contain" in html
    assert "toolName === 'GenerateVideo'" in html
    assert "brief = resultText.split('\\n')[0]" in html


@pytest.mark.asyncio
async def test_bearer_video_client_rewrite_and_asset_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any):
            calls.append({"method": method, "url": url, **kwargs})
            if method == "POST" and url.endswith("/prompt-rewrite"):
                payload = {"task_id": "rw-1", "status": "queued"}
            elif method == "GET" and url.endswith("/prompt-rewrite/rw-1"):
                payload = {"task_id": "rw-1", "status": "completed", "prompt": "改写后提示词"}
            elif method == "POST" and url.endswith("/assets/groups"):
                payload = {"group_id": "group-1", "client_id": "client"}
            else:
                payload = {"ok": True}
            return types.SimpleNamespace(status_code=200, text=json.dumps(payload), json=lambda: payload)

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))
    client = BearerVideoGenerationClient(base_url="https://api.video.test", token="token", poll_interval=0)

    rewritten = await client.rewrite_prompt(prompt="做一条耳机广告", pe_mode="creative", duration=8)
    group = await client.create_asset_group(name="avatars", description="虚拟人像")

    assert rewritten["prompt"] == "改写后提示词"
    assert group["group_id"] == "group-1"
    assert calls[0]["json"]["pe_mode"] == "creative"
    assert calls[0]["headers"]["Authorization"] == "Bearer token"
    assert calls[2]["json"] == {"name": "avatars", "description": "虚拟人像"}


class FakeVideoApiClient:
    async def rewrite_prompt(self, **kwargs: Any) -> dict[str, Any]:
        return {"status": "completed", "prompt": "rewritten", "kwargs": kwargs}

    async def upload_asset(self, **kwargs: Any) -> dict[str, Any]:
        return {"asset_id": "asset-1", "asset_uri": "asset://asset-1", "kwargs": kwargs}


class FailingVideoApiClient(FakeVideoApiClient):
    async def rewrite_prompt(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("rewrite failed: upstream payload={\"code\":400,\"message\":\"invalid pe_mode\"}")


@pytest.mark.asyncio
async def test_video_api_tool_dispatches_rewrite_and_asset_upload() -> None:
    tool = VideoApiTool(lambda: FakeVideoApiClient())

    rewrite = await tool.call(
        {"action": "rewrite_prompt", "prompt": "广告", "pe_mode": "reference", "duration": 6, "images": [{"url": "asset://asset-1", "role": "reference_image"}]},
        ToolContext(session_id="s", messages=[]),
    )
    upload = await tool.call(
        {"action": "upload_asset", "group_id": "group-1", "url": "https://example.com/a.png", "name": "avatar", "asset_type": "Image"},
        ToolContext(session_id="s", messages=[]),
    )

    assert rewrite.is_error is False
    assert rewrite.metadata["response"]["prompt"] == "rewritten"
    assert upload.is_error is False
    assert upload.metadata["response"]["asset_uri"] == "asset://asset-1"


@pytest.mark.asyncio
async def test_video_api_tool_returns_full_dispatch_exception_to_agent() -> None:
    tool = VideoApiTool(lambda: FailingVideoApiClient())

    result = await tool.call(
        {"action": "rewrite_prompt", "prompt": "广告", "pe_mode": "storyboard"},
        ToolContext(session_id="s", messages=[]),
    )

    assert result.is_error is True
    assert "VideoApi rewrite_prompt failed" in result.content
    assert "ValueError: rewrite failed" in result.content
    assert "invalid pe_mode" in result.content
    assert "Traceback:" in result.content
    assert result.metadata["error_type"] == "ValueError"
