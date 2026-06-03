from __future__ import annotations

import io
import logging
from typing import Any, Callable

from agent_core.attachments import ImageAttachmentStore
from agent_core.images import ImageGenerationClient, ImageInput
from agent_core.tools.base import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# Images larger than this (in bytes) will be auto-compressed before sending to the API.
_MAX_IMAGE_BYTES = 512 * 1024  # 512 KB
_COMPRESS_QUALITY = 75
_COMPRESS_MAX_DIM = 2048


class GenerateImageTool:
    name = "GenerateImage"
    description = "Generate or edit images as session attachments using the configured image-generation provider. This is a tool capability and does not affect the chat model provider."
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Image prompt or edit instruction."},
            "mode": {"type": "string", "enum": ["generate", "edit"], "description": "Use generate for new images, edit to modify existing image attachments."},
            "size": {"type": "string", "description": "Image size, e.g. 1024x1024, 1536x1024, 1024x1536, or auto."},
            "quality": {"type": "string", "description": "Optional quality value: auto, low, medium, high."},
            "output_format": {"type": "string", "description": "png, jpeg, or webp."},
            "n": {"type": "integer", "minimum": 1, "maximum": 4, "description": "Number of images to generate."},
            "input_attachment_ids": {"type": "array", "items": {"type": "string"}, "description": "Existing image attachment IDs used as edit references."},
        },
        "required": ["prompt"],
    }
    is_concurrency_safe = True
    concurrency_group = "image_generation"
    max_concurrency = 3
    should_defer = False

    def __init__(self, attachment_store: ImageAttachmentStore, client_provider: Callable[[], ImageGenerationClient | None]) -> None:
        self._attachment_store = attachment_store
        self._client_provider = client_provider

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        client = self._client_provider()
        if client is None:
            return ToolResult(
                content="GenerateImage failed: image generation is not configured. Configure AGENT_IMAGE_* or Codex image credentials first.",
                is_error=True,
            )
        prompt = str(tool_input.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(content="GenerateImage failed: prompt is required", is_error=True)
        mode = str(tool_input.get("mode") or "generate").strip().lower()
        size = str(tool_input.get("size") or "1024x1024")
        quality = str(tool_input.get("quality") or "").strip() or None
        output_format = str(tool_input.get("output_format") or "png").strip().lower()
        n = max(1, min(int(tool_input.get("n") or 1), 4))
        user_id = str(context.metadata.get("user_id") or "")
        org_id = str(context.metadata.get("org_id") or "")
        input_ids = [str(x) for x in (tool_input.get("input_attachment_ids") or []) if str(x).strip()]

        input_images: list[ImageInput] = []
        for attachment_id in input_ids[:16]:
            item = self._attachment_store.get_authorized(
                attachment_id=attachment_id,
                user_id=user_id,
                org_id=org_id,
                session_id=context.session_id,
            )
            if item is None:
                return ToolResult(content=f"GenerateImage failed: input attachment not found: {attachment_id}", is_error=True)
            if not item.content_type.startswith("image/"):
                return ToolResult(content=f"GenerateImage failed: input attachment must be an image: {attachment_id}", is_error=True)
            raw_data = self._attachment_store.read_bytes(item)
            compressed_data, compressed_ct = _compress_if_needed(raw_data, item.content_type)
            input_images.append(ImageInput(data=compressed_data, content_type=compressed_ct, filename=item.filename))

        if mode == "edit":
            if not input_images:
                return ToolResult(content="GenerateImage failed: edit mode requires input_attachment_ids", is_error=True)
            result = await client.edit_image(prompt=prompt, input_images=input_images, size=size, quality=quality, output_format=output_format, n=n)
        else:
            result = await client.generate_image(prompt=prompt, size=size, quality=quality, output_format=output_format, n=n)

        saved = []
        for idx, image in enumerate(result.images, 1):
            item = self._attachment_store.save_media(
                data=image.data,
                filename=f"agent-image-{mode}-{idx}.{_extension_for_content_type(image.content_type)}",
                content_type=image.content_type,
                user_id=user_id,
                org_id=org_id,
                session_id=context.session_id,
                metadata={"source": "GenerateImage", "prompt": prompt, "mode": mode, "revised_prompt": image.revised_prompt, "raw_url": image.raw_url},
            )
            saved.append(item.to_dict())
        lines = ["GenerateImage completed. Saved image attachment IDs:"]
        lines.extend(f"- {item['id']} ({item.get('filename')})" for item in saved)
        return ToolResult(content="\n".join(lines), metadata={"attachments": saved})


def _compress_if_needed(data: bytes, content_type: str) -> tuple[bytes, str]:
    """Compress image if it exceeds _MAX_IMAGE_BYTES. Returns (data, content_type)."""
    if len(data) <= _MAX_IMAGE_BYTES:
        return data, content_type

    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available, cannot auto-compress large input image")
        return data, content_type

    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) > _COMPRESS_MAX_DIM:
            ratio = _COMPRESS_MAX_DIM / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        out = io.BytesIO()
        fmt = "JPEG" if content_type in ("image/jpeg", "image/jpg") else "JPEG"
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out, format=fmt, quality=_COMPRESS_QUALITY, optimize=True)
        compressed = out.getvalue()
        logger.info("Compressed input image: %d -> %d bytes", len(data), len(compressed))
        return compressed, "image/jpeg"
    except Exception:
        logger.warning("Failed to compress input image, sending as-is", exc_info=True)
        return data, content_type


def _extension_for_content_type(content_type: str) -> str:
    if content_type == "image/jpeg":
        return "jpg"
    if content_type == "image/webp":
        return "webp"
    if content_type == "image/gif":
        return "gif"
    return "png"
