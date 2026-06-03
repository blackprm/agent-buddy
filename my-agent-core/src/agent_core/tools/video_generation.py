from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Callable

from agent_core.attachments import ImageAttachmentStore
from agent_core.attachments.store import ImageAttachment
from agent_core.core.events import AgentEvent
from agent_core.recovery.tool_errors import format_exception_detail
from agent_core.tools.base import ToolContext, ToolResult
from agent_core.videos import VideoGenerationClient, VideoInput
from agent_core.videos.rest_bearer import image_bytes_to_data_url
from agent_core.videos.tos_store import TosMediaStore, is_tos_configured


class GenerateVideoTool:
    name = "GenerateVideo"
    description = "Generate a video as a session attachment using the configured video-generation provider. Supports text-to-video, image/video/audio session attachments, TOS-backed signed reference URLs, asset:// references, direct media URLs, and creative prompt enhancement parameters. This tool does not affect the chat model provider."
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Video prompt. Quote dialogue explicitly when speech is needed."},
            "mode": {"type": "string", "enum": ["fast", "pro"], "description": "fast for lower cost/speed, pro for highest quality and multimodal references."},
            "ratio": {"type": "string", "enum": ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"]},
            "duration": {"type": "integer", "description": "Video duration in seconds. fast: 4-12, pro: 4-15, or -1 for automatic."},
            "resolution": {"type": "string", "enum": ["480p", "720p"]},
            "generate_audio": {"type": "boolean", "description": "Whether to generate synchronized audio."},
            "watermark": {"type": "boolean"},
            "pe_mode": {"type": "string", "enum": ["", "hot_v_creative", "storyboard", "creative", "reference", "auto"], "description": "Optional prompt enhancement mode."},
            "rewrite_thinking_level": {"type": "string", "enum": ["standard", "accelerated"]},
            "web_search": {"type": "boolean"},
            "auto_queued": {"type": "boolean"},
            "seed": {"type": "integer"},
            "image_attachment_ids": {"type": "array", "items": {"type": "string"}, "description": "Session image attachments to pass as base64 image references."},
            "image_urls": {"type": "array", "items": {"type": "string"}, "description": "Image URLs, base64 data URLs, or asset:// IDs."},
            "image_role": {"type": "string", "enum": ["first_frame", "last_frame", "reference_image"]},
            "video_attachment_ids": {"type": "array", "items": {"type": "string"}, "description": "Session video attachments uploaded to TOS and passed as signed reference_video URLs."},
            "video_urls": {"type": "array", "items": {"type": "string"}, "description": "Video URLs or asset:// IDs used as reference_video."},
            "video_role": {"type": "string", "enum": ["reference_video"]},
            "audio_attachment_ids": {"type": "array", "items": {"type": "string"}, "description": "Session audio attachments uploaded to TOS and passed as signed reference_audio URLs."},
            "audio_urls": {"type": "array", "items": {"type": "string"}, "description": "Audio URLs or asset:// IDs used as reference_audio."},
            "audio_role": {"type": "string", "enum": ["reference_audio"]},
            "video_payload": {"type": "object", "description": "Advanced raw payload fields merged into /create request."},
        },
        "required": ["prompt"],
    }
    is_concurrency_safe = True
    concurrency_group = "video_generation"
    max_concurrency = 3
    should_defer = False

    def __init__(self, attachment_store: ImageAttachmentStore, client_provider: Callable[[], VideoGenerationClient | None], tos_store: TosMediaStore | None = None, *, heartbeat_interval: float = 30.0) -> None:
        self._attachment_store = attachment_store
        self._client_provider = client_provider
        self._tos_store = tos_store
        self._tos_store_checked = tos_store is not None
        self._heartbeat_interval = max(0.001, float(heartbeat_interval or 30.0))

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        client = self._client_provider()
        if client is None:
            return ToolResult(content="GenerateVideo failed: video generation is not configured. Configure AGENT_VIDEO_BASE_URL and AGENT_VIDEO_TOKEN first.", is_error=True)
        prompt = str(tool_input.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(content="GenerateVideo failed: prompt is required", is_error=True)

        user_id = str(context.metadata.get("user_id") or "")
        org_id = str(context.metadata.get("org_id") or "")
        image_role = str(tool_input.get("image_role") or "reference_image").strip() or "reference_image"
        video_role = str(tool_input.get("video_role") or "reference_video").strip() or "reference_video"
        audio_role = str(tool_input.get("audio_role") or "reference_audio").strip() or "reference_audio"
        uploaded_tos_assets: list[dict[str, Any]] = []
        try:
            images = await self._url_inputs(_string_list(tool_input.get("image_urls")), role=image_role)
            for attachment_id in _string_list(tool_input.get("image_attachment_ids"))[:9]:
                images.append(
                    await self._attachment_input(
                        attachment_id=attachment_id,
                        expected_prefix="image/",
                        ids_field="image_attachment_ids",
                        role=image_role,
                        media_kind="image",
                        user_id=user_id,
                        org_id=org_id,
                        session_id=context.session_id,
                        require_tos=False,
                        uploaded=uploaded_tos_assets,
                    )
                )

            videos = await self._url_inputs(_string_list(tool_input.get("video_urls")), role=video_role)
            for attachment_id in _string_list(tool_input.get("video_attachment_ids"))[:4]:
                videos.append(
                    await self._attachment_input(
                        attachment_id=attachment_id,
                        expected_prefix="video/",
                        ids_field="video_attachment_ids",
                        role=video_role,
                        media_kind="video",
                        user_id=user_id,
                        org_id=org_id,
                        session_id=context.session_id,
                        require_tos=True,
                        uploaded=uploaded_tos_assets,
                    )
                )

            audios = await self._url_inputs(_string_list(tool_input.get("audio_urls")), role=audio_role)
            for attachment_id in _string_list(tool_input.get("audio_attachment_ids"))[:4]:
                audios.append(
                    await self._attachment_input(
                        attachment_id=attachment_id,
                        expected_prefix="audio/",
                        ids_field="audio_attachment_ids",
                        role=audio_role,
                        media_kind="audio",
                        user_id=user_id,
                        org_id=org_id,
                        session_id=context.session_id,
                        require_tos=True,
                        uploaded=uploaded_tos_assets,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult(content=f"GenerateVideo failed while preparing media references:\n{format_exception_detail(exc)}", is_error=True, metadata={"error_type": type(exc).__name__})

        metadata = {key: tool_input[key] for key in ("pe_mode", "rewrite_thinking_level", "web_search", "auto_queued", "seed", "video_payload") if key in tool_input and tool_input[key] not in (None, "")}
        if uploaded_tos_assets:
            metadata["uploaded_tos_assets"] = uploaded_tos_assets

        try:
            result = await self._generate_with_heartbeat(
                client,
                context,
                prompt=prompt,
                kwargs={
                    "prompt": prompt,
                    "mode": str(tool_input.get("mode") or "fast"),
                    "ratio": str(tool_input.get("ratio") or "adaptive"),
                    "duration": int(tool_input.get("duration") or 5),
                    "resolution": str(tool_input.get("resolution") or "720p"),
                    "generate_audio": bool(tool_input.get("generate_audio", True)),
                    "watermark": bool(tool_input.get("watermark", False)),
                    "images": images,
                    "videos": videos,
                    "audios": audios,
                    "metadata": metadata,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult(content=f"GenerateVideo failed during provider request or result download:\n{format_exception_detail(exc)}", is_error=True, metadata={"error_type": type(exc).__name__})

        saved = []
        for idx, video in enumerate(result.videos, 1):
            item = self._attachment_store.save_media(
                data=video.data,
                filename=f"agent-video-{idx}.{_extension_for_content_type(video.content_type)}",
                content_type=video.content_type,
                user_id=user_id,
                org_id=org_id,
                session_id=context.session_id,
                metadata={"source": "GenerateVideo", "prompt": prompt, "task_id": video.task_id, "raw_url": video.raw_url, "video_metadata": video.metadata, "generation_metadata": {k: v for k, v in result.metadata.items() if k != "raw_response"}},
            )
            saved.append(item.to_dict())
        lines = ["GenerateVideo completed. Saved video attachment IDs:"]
        lines.extend(f"- {item['id']} ({item.get('filename')})" for item in saved)
        return ToolResult(content="\n".join(lines), metadata={"attachments": saved, "generation": {k: v for k, v in result.metadata.items() if k != "raw_response"}})

    async def _url_inputs(self, urls: list[str], *, role: str) -> list[VideoInput]:
        inputs: list[VideoInput] = []
        for url in urls:
            inputs.append(VideoInput(url=await self._resolve_reference_url(url), role=role))
        return inputs

    async def _resolve_reference_url(self, url: str) -> str:
        if not url.startswith("tos://"):
            return url
        tos_store = self._get_tos_store()
        if tos_store is None:
            raise ValueError("tos:// media references require TOS configuration")
        key = tos_store.parse_tos_uri(url)
        if not key:
            raise ValueError("invalid TOS media reference")
        return await tos_store.presign_key(key)

    async def _attachment_input(
        self,
        *,
        attachment_id: str,
        expected_prefix: str,
        ids_field: str,
        role: str,
        media_kind: str,
        user_id: str,
        org_id: str,
        session_id: str,
        require_tos: bool,
        uploaded: list[dict[str, Any]],
    ) -> VideoInput:
        item = self._attachment_store.get_authorized(attachment_id=attachment_id, user_id=user_id, org_id=org_id, session_id=session_id)
        if item is None:
            raise ValueError(f"{media_kind} attachment not found: {attachment_id}")
        if not item.content_type.startswith(expected_prefix):
            raise ValueError(f"attachment must be a {expected_prefix.rstrip('/')} for {ids_field}: {attachment_id}")
        tos_store = self._get_tos_store()
        data = self._attachment_store.read_bytes(item)
        if tos_store is None:
            if require_tos:
                raise ValueError(f"{ids_field} requires TOS configuration because video/audio references must be reachable by the video API")
            return VideoInput(url=image_bytes_to_data_url(data, item.content_type), role=role)
        uploaded_obj = await tos_store.upload_bytes(
            data=data,
            filename=item.filename,
            content_type=item.content_type,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            media_kind=media_kind,
        )
        uploaded.append(_tos_asset_metadata(item, media_kind, uploaded_obj.key, uploaded_obj.stored_url, uploaded_obj.expires))
        return VideoInput(url=uploaded_obj.signed_url, role=role)

    def _get_tos_store(self) -> TosMediaStore | None:
        if self._tos_store is not None:
            return self._tos_store
        if self._tos_store_checked:
            return None
        self._tos_store_checked = True
        if not is_tos_configured():
            return None
        self._tos_store = TosMediaStore()
        return self._tos_store

    async def _generate_with_heartbeat(self, client: VideoGenerationClient, context: ToolContext, *, prompt: str, kwargs: dict[str, Any]):
        if context.event_callback is None:
            return await client.generate_video(**kwargs)
        started_at = time.monotonic()
        task = asyncio.create_task(client.generate_video(**kwargs), name="GenerateVideo-client")
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self._heartbeat_interval)
                if task in done:
                    return task.result()
                elapsed = int(time.monotonic() - started_at)
                await context.event_callback(AgentEvent("tool_progress", {
                    "tool_use_id": context.metadata.get("tool_use_id", ""),
                    "tool": "GenerateVideo",
                    "status": "waiting",
                    "message": "Video generation is still running",
                    "elapsedTimeSeconds": elapsed,
                    "promptPreview": prompt[:120],
                }))
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extension_for_content_type(content_type: str) -> str:
    if content_type == "video/quicktime":
        return "mov"
    if content_type == "video/webm":
        return "webm"
    if content_type == "video/mpeg":
        return "mpeg"
    if content_type == "video/x-msvideo":
        return "avi"
    return "mp4"


def _tos_asset_metadata(item: ImageAttachment, media_kind: str, key: str, stored_url: str, expires: int) -> dict[str, Any]:
    return {
        "attachment_id": item.id,
        "media_kind": media_kind,
        "content_type": item.content_type,
        "filename": item.filename,
        "tos_key": key,
        "stored_url": stored_url,
        "signed_url_expires": expires,
    }
