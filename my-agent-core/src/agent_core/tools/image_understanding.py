from __future__ import annotations

from typing import Any

from agent_core.attachments import ImageAttachmentStore
from agent_core.tools.base import ToolContext, ToolResult
from agent_core.vision.base import VisionClient, VisionMediaInput


DEFAULT_IMAGE_PROMPT = (
    "Please understand this image for the current conversation. Describe the visible content, "
    "extract any important text, and call out details relevant to the user's request."
)

DEFAULT_VIDEO_PROMPT = (
    "Please understand this video for the current conversation. Summarize the visible content, "
    "important timeline changes, readable text, and details relevant to the user's request."
)

DEFAULT_MEDIA_PROMPT = (
    "Please understand these images and videos together for the current conversation. Describe the visible content, "
    "summarize important timeline changes, extract readable text, and call out details relevant to the user's request."
)


class UnderstandImageTool:
    name = "UnderstandImage"
    description = (
        "Understand one or more images/videos uploaded by the user, including mixed image+video inputs. "
        "Use this when the user references attached images, screenshots, diagrams, photos, videos, or asks what "
        "is in media. The input must be an attachment_id or attachment_ids from the user's message attachment list."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {"type": "string", "description": "Media attachment id, e.g. img_... or vid_..."},
            "attachment_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more media attachment ids, e.g. [img_..., vid_...] for mixed image+video understanding.",
            },
            "prompt": {"type": "string", "description": "Optional focused question or instruction for media understanding"},
        },
        "anyOf": [{"required": ["attachment_id"]}, {"required": ["attachment_ids"]}],
    }
    is_concurrency_safe = True
    should_defer = False

    def __init__(self, image_store: ImageAttachmentStore, vision_client: VisionClient | None) -> None:
        self._image_store = image_store
        self._vision_client = vision_client

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if self._vision_client is None:
            return ToolResult(
                content="UnderstandImage is not configured: set a vision model/API key in the server environment.",
                is_error=True,
            )
        attachment_ids = self._attachment_ids(tool_input)
        if not attachment_ids:
            return ToolResult(content="attachment_id or attachment_ids is required", is_error=True)
        user_id = str(context.metadata.get("user_id") or "")
        org_id = str(context.metadata.get("org_id") or "")
        items = []
        media_inputs: list[VisionMediaInput] = []
        for attachment_id in attachment_ids:
            item = self._image_store.get_authorized(
                attachment_id=attachment_id,
                user_id=user_id,
                org_id=org_id,
                session_id=context.session_id,
            )
            if item is None:
                return ToolResult(content=f"Media attachment not found or not accessible: {attachment_id}", is_error=True)
            if not (item.content_type.startswith("image/") or item.content_type.startswith("video/")):
                return ToolResult(
                    content=f"Unsupported media type for vision understanding: {item.content_type} ({attachment_id})",
                    is_error=True,
                )
            items.append(item)
            media_inputs.append(
                VisionMediaInput(
                    data=self._image_store.read_bytes(item),
                    content_type=item.content_type,
                    filename=item.filename,
                    attachment_id=item.id,
                )
            )

        item = items[0]
        media_bytes = media_inputs[0].data
        prompt = str(tool_input.get("prompt") or "")
        if len(media_inputs) > 1:
            result = await self._vision_client.understand_media(
                media=media_inputs,
                prompt=prompt or DEFAULT_MEDIA_PROMPT,
                metadata={"attachment_ids": [media.attachment_id for media in media_inputs], "session_id": context.session_id},
            )
        elif item.content_type.startswith("video/"):
            result = await self._vision_client.understand_video(
                video_bytes=media_bytes,
                content_type=item.content_type,
                prompt=prompt or DEFAULT_VIDEO_PROMPT,
                metadata={"attachment_id": item.id, "session_id": context.session_id},
            )
        else:
            result = await self._vision_client.understand_image(
                image_bytes=media_bytes,
                content_type=item.content_type,
                prompt=prompt or DEFAULT_IMAGE_PROMPT,
                metadata={"attachment_id": item.id, "session_id": context.session_id},
            )
        return ToolResult(
            content=result.text,
            metadata={
                "attachment": item.to_dict(),
                "attachments": [item.to_dict() for item in items],
                "vision": result.metadata,
            },
        )

    @staticmethod
    def _attachment_ids(tool_input: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        single_id = str(tool_input.get("attachment_id") or "").strip()
        if single_id:
            ids.append(single_id)
        raw_ids = tool_input.get("attachment_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        for value in raw_ids:
            attachment_id = str(value or "").strip()
            if attachment_id and attachment_id not in ids:
                ids.append(attachment_id)
        return ids
