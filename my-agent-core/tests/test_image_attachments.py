from __future__ import annotations

import base64
import io
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from agent_core.attachments import ImageAttachmentStore
from agent_core.context.builder import ContextBuilder
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.model.base import ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_core.tools.base import ToolContext
from agent_core.tools.base import ToolRegistry
from agent_core.tools.image_understanding import UnderstandImageTool
from agent_core.types import TextBlock
from agent_core.vision.openai_compatible import OpenAICompatibleVisionClient
from agent_core.vision.base import VisionMediaInput, VisionResult
from agent_server.app import _content_disposition_inline, _parse_multipart_file


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"0" * 32
AUDIO_BYTES = b"ID3" + b"0" * 32


def _data_url_bytes(url: str) -> tuple[str, bytes]:
    header, encoded = url.split(",", 1)
    return header, base64.b64decode(encoded)


def _large_noise_png() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    image = Image.effect_noise((900, 900), 100).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_runtime_persists_uploaded_media_turn_before_model_finishes(tmp_path: Path) -> None:
    session_store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    runtime = AgentRuntime(
        model=ScriptedModelClient([ModelResponse(content=[TextBlock("ok")], stop_reason="end_turn")]),
        tools=ToolRegistry([]),
        session_store=session_store,
        context_builder=ContextBuilder(base_instructions="test"),
        config=AgentRuntimeConfig(session_id="session-1", user_id="user-1", org_id="org-1"),
    )
    events = runtime.run(
        "看一下这张图",
        attachments=[{"id": "img_1", "filename": "screen.png", "content_type": "image/png", "size_bytes": len(PNG_BYTES)}],
    )

    first = await anext(events)

    assert first.type == "loop_started"
    saved = session_store.load_messages("session-1")
    assert saved[0].metadata["original_user_input"] == "看一下这张图"
    assert saved[0].metadata["attachments"][0]["id"] == "img_1"
    await events.aclose()


def test_image_attachment_store_saves_and_authorizes_by_scope(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)

    item = store.save_image(
        data=PNG_BYTES,
        filename="screen.png",
        content_type="image/png",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )

    assert item.id.startswith("img_")
    assert item.content_type == "image/png"
    assert item.size_bytes == len(PNG_BYTES)
    assert store.read_bytes(item) == PNG_BYTES
    assert store.get_authorized(
        attachment_id=item.id,
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    ) is not None
    assert store.get_authorized(
        attachment_id=item.id,
        user_id="user-2",
        org_id="org-1",
        session_id="session-1",
    ) is None


def test_image_attachment_store_rejects_unsupported_types(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)

    with pytest.raises(ValueError, match="unsupported media content type"):
        store.save_image(
            data=b"not an image",
            filename="notes.txt",
            content_type="text/plain",
            user_id="user-1",
            org_id="org-1",
            session_id="session-1",
        )


def test_image_attachment_store_detects_png_with_filetype_when_content_type_is_generic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_filetype = types.SimpleNamespace(
        guess=lambda _data: types.SimpleNamespace(mime="image/png"),
    )
    monkeypatch.setitem(sys.modules, "filetype", fake_filetype)
    store = ImageAttachmentStore(tmp_path)

    item = store.save_image(
        data=PNG_BYTES,
        filename="blob",
        content_type="application/octet-stream",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )

    assert item.content_type == "image/png"


def test_attachment_store_saves_video(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)

    item = store.save_media(
        data=MP4_BYTES,
        filename="clip.mp4",
        content_type="video/mp4",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )

    assert item.id.startswith("vid_")
    assert item.content_type == "video/mp4"
    assert store.read_bytes(item) == MP4_BYTES


def test_parse_multipart_file_without_python_multipart_dependency() -> None:
    boundary = "----agent-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="paste.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + PNG_BYTES + f"\r\n--{boundary}--\r\n".encode("utf-8")

    filename, content_type, data = _parse_multipart_file(
        body,
        f"multipart/form-data; boundary={boundary}",
    )

    assert filename == "paste.png"
    assert content_type == "image/png"
    assert data == PNG_BYTES


def test_content_disposition_inline_encodes_non_ascii_filename() -> None:
    disposition = _content_disposition_inline("小红书参考图.png")

    disposition.encode("latin-1")
    assert 'filename="attachment.png"' in disposition
    assert "filename*=UTF-8''%E5%B0%8F%E7%BA%A2%E4%B9%A6%E5%8F%82%E8%80%83%E5%9B%BE.png" in disposition


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def understand_image(self, **kwargs: Any) -> VisionResult:
        self.calls.append(kwargs)
        return VisionResult(text="The image contains a tiny test PNG.", metadata={"model": "fake-vision"})

    async def understand_video(self, **kwargs: Any) -> VisionResult:
        self.calls.append(kwargs)
        return VisionResult(text="The video contains a tiny test MP4.", metadata={"model": "fake-vision"})

    async def understand_media(self, **kwargs: Any) -> VisionResult:
        self.calls.append(kwargs)
        media = kwargs["media"]
        assert all(isinstance(item, VisionMediaInput) for item in media)
        return VisionResult(text=f"Understood {len(media)} media files together.", metadata={"model": "fake-vision"})


class RecordingOpenAICompatibleVisionClient(OpenAICompatibleVisionClient):
    def __init__(self) -> None:
        self.content: list[dict[str, Any]] | None = None

    async def _understand_content(self, *, content: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> VisionResult:
        self.content = content
        return VisionResult(text="ok", metadata={})


@pytest.mark.asyncio
async def test_understand_image_tool_reads_authorized_attachment(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    item = store.save_image(
        data=PNG_BYTES,
        filename="screen.png",
        content_type="image/png",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    vision = FakeVisionClient()
    tool = UnderstandImageTool(store, vision)

    result = await tool.call(
        {"attachment_id": item.id, "prompt": "What is this?"},
        ToolContext(
            session_id="session-1",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1"},
            cwd=str(tmp_path),
        ),
    )

    assert result.is_error is False
    assert "tiny test PNG" in result.content
    assert vision.calls[0]["image_bytes"] == PNG_BYTES
    assert vision.calls[0]["content_type"] == "image/png"
    assert result.metadata["attachment"]["id"] == item.id


@pytest.mark.asyncio
async def test_understand_image_tool_denies_cross_session_access(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    item = store.save_image(
        data=PNG_BYTES,
        filename="screen.png",
        content_type="image/png",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    tool = UnderstandImageTool(store, FakeVisionClient())

    result = await tool.call(
        {"attachment_id": item.id},
        ToolContext(
            session_id="session-2",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1"},
            cwd=str(tmp_path),
        ),
    )

    assert result.is_error is True
    assert "not found or not accessible" in result.content


@pytest.mark.asyncio
async def test_understand_image_tool_reads_authorized_video(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    item = store.save_media(
        data=MP4_BYTES,
        filename="clip.mp4",
        content_type="video/mp4",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    vision = FakeVisionClient()
    tool = UnderstandImageTool(store, vision)

    result = await tool.call(
        {"attachment_id": item.id, "prompt": "What happens?"},
        ToolContext(
            session_id="session-1",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1"},
            cwd=str(tmp_path),
        ),
    )

    assert result.is_error is False
    assert "tiny test MP4" in result.content
    assert vision.calls[0]["video_bytes"] == MP4_BYTES
    assert vision.calls[0]["content_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_understand_image_tool_reads_mixed_image_and_video(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    image = store.save_image(
        data=PNG_BYTES,
        filename="screen.png",
        content_type="image/png",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    video = store.save_media(
        data=MP4_BYTES,
        filename="clip.mp4",
        content_type="video/mp4",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    vision = FakeVisionClient()
    tool = UnderstandImageTool(store, vision)

    result = await tool.call(
        {"attachment_ids": [image.id, video.id], "prompt": "Compare these."},
        ToolContext(
            session_id="session-1",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1"},
            cwd=str(tmp_path),
        ),
    )

    assert result.is_error is False
    assert "2 media files" in result.content
    assert len(vision.calls) == 1
    call = vision.calls[0]
    assert call["prompt"] == "Compare these."
    assert call["metadata"]["attachment_ids"] == [image.id, video.id]
    assert [(item.attachment_id, item.content_type, item.data) for item in call["media"]] == [
        (image.id, "image/png", PNG_BYTES),
        (video.id, "video/mp4", MP4_BYTES),
    ]
    assert [item["id"] for item in result.metadata["attachments"]] == [image.id, video.id]


@pytest.mark.asyncio
async def test_understand_image_tool_rejects_non_visual_media(tmp_path: Path) -> None:
    store = ImageAttachmentStore(tmp_path)
    audio = store.save_media(
        data=AUDIO_BYTES,
        filename="sound.mp3",
        content_type="audio/mpeg",
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
    )
    vision = FakeVisionClient()
    tool = UnderstandImageTool(store, vision)

    result = await tool.call(
        {"attachment_ids": [audio.id], "prompt": "What is this?"},
        ToolContext(
            session_id="session-1",
            messages=[],
            metadata={"user_id": "user-1", "org_id": "org-1"},
            cwd=str(tmp_path),
        ),
    )

    assert result.is_error is True
    assert "Unsupported media type" in result.content
    assert vision.calls == []


@pytest.mark.asyncio
async def test_openai_compatible_vision_client_builds_mixed_media_content() -> None:
    client = RecordingOpenAICompatibleVisionClient()

    result = await client.understand_media(
        media=[
            VisionMediaInput(data=PNG_BYTES, content_type="image/png", filename="screen.png", attachment_id="img_1"),
            VisionMediaInput(data=MP4_BYTES, content_type="video/mp4", filename="clip.mp4", attachment_id="vid_1"),
        ],
        prompt="Analyze both.",
    )

    assert result.text == "ok"
    assert client.content is not None
    assert [part["type"] for part in client.content] == ["text", "image_url", "video_url"]
    assert client.content[0] == {"type": "text", "text": "Analyze both."}
    assert client.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert client.content[2]["video_url"]["url"].startswith("data:video/mp4;base64,")


@pytest.mark.asyncio
async def test_openai_compatible_vision_client_compresses_large_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VISION_MAX_INPUT_IMAGE_BYTES", str(256 * 1024))
    image_bytes = _large_noise_png()
    client = RecordingOpenAICompatibleVisionClient()

    result = await client.understand_image(
        image_bytes=image_bytes,
        content_type="image/png",
        prompt="Analyze this oversized screenshot.",
    )

    assert result.text == "ok"
    assert client.content is not None
    assert client.content[0] == {"type": "text", "text": "Analyze this oversized screenshot."}
    url = client.content[1]["image_url"]["url"]
    header, compressed = _data_url_bytes(url)
    assert header == "data:image/jpeg;base64"
    assert len(compressed) <= 256 * 1024
    assert len(compressed) < len(image_bytes)


@pytest.mark.asyncio
async def test_openai_compatible_vision_client_compresses_large_image_in_mixed_media(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VISION_MAX_INPUT_IMAGE_BYTES", str(256 * 1024))
    image_bytes = _large_noise_png()
    client = RecordingOpenAICompatibleVisionClient()

    result = await client.understand_media(
        media=[
            VisionMediaInput(data=image_bytes, content_type="image/png", filename="screen.png", attachment_id="img_1"),
            VisionMediaInput(data=MP4_BYTES, content_type="video/mp4", filename="clip.mp4", attachment_id="vid_1"),
        ],
        prompt="Analyze both.",
    )

    assert result.text == "ok"
    assert client.content is not None
    image_url = client.content[1]["image_url"]["url"]
    video_url = client.content[2]["video_url"]["url"]
    image_header, compressed = _data_url_bytes(image_url)
    video_header, video_bytes = _data_url_bytes(video_url)
    assert image_header == "data:image/jpeg;base64"
    assert len(compressed) <= 256 * 1024
    assert len(compressed) < len(image_bytes)
    assert video_header == "data:video/mp4;base64"
    assert video_bytes == MP4_BYTES
