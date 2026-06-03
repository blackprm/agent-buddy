from __future__ import annotations

import base64
import io
import os
from typing import Any

from agent_core.vision.base import VisionMediaInput, VisionResult


_DEFAULT_MAX_VISION_IMAGE_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_VISION_IMAGE_DIMENSION = 2048
_DEFAULT_VISION_IMAGE_QUALITY = 85


class OpenAICompatibleVisionClient:
    """Small vision client for OpenAI-compatible chat completion APIs."""

    def __init__(self, *, model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 2048) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens

    async def understand_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        image_bytes, content_type = _compress_image_for_vision_if_needed(image_bytes, content_type)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return await self._understand_media(
            media_type="image_url",
            media_payload={"url": f"data:{content_type};base64,{image_b64}"},
            prompt=prompt or "Describe and analyze this image.",
            metadata=metadata,
        )

    async def understand_video(
        self,
        *,
        video_bytes: bytes,
        content_type: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        video_b64 = base64.b64encode(video_bytes).decode("ascii")
        return await self._understand_media(
            media_type="video_url",
            media_payload={"url": f"data:{content_type};base64,{video_b64}"},
            prompt=prompt or "Describe and analyze this video.",
            metadata=metadata,
        )

    async def understand_media(
        self,
        *,
        media: list[VisionMediaInput],
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt or "Describe and analyze these media files together."}]
        for item in media:
            if item.content_type.startswith("video/"):
                encoded = base64.b64encode(item.data).decode("ascii")
                url = f"data:{item.content_type};base64,{encoded}"
                content.append({"type": "video_url", "video_url": {"url": url}})
            elif item.content_type.startswith("image/"):
                image_bytes, content_type = _compress_image_for_vision_if_needed(item.data, item.content_type)
                encoded = base64.b64encode(image_bytes).decode("ascii")
                url = f"data:{content_type};base64,{encoded}"
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                raise ValueError(f"unsupported vision media content type: {item.content_type}")
        return await self._understand_content(content=content, metadata=metadata)

    async def _understand_media(
        self,
        *,
        media_type: str,
        media_payload: dict[str, Any],
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        return await self._understand_content(
            content=[
                {"type": "text", "text": prompt},
                {"type": media_type, media_type: media_payload},
            ],
            metadata=metadata,
        )

    async def _understand_content(
        self,
        *,
        content: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=int((metadata or {}).get("max_tokens") or self._max_tokens),
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        choice = response.choices[0] if response.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            text = str(choice.message.content)
        usage = getattr(response, "usage", None)
        usage_data = {}
        if usage:
            usage_data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        return VisionResult(text=text, metadata={"model": self._model, "usage": usage_data})


def _compress_image_for_vision_if_needed(image_bytes: bytes, content_type: str) -> tuple[bytes, str]:
    """Keep inline vision image data URLs under gateway/model limits before base64 encoding.

    Vision requests inline image bytes as base64 data URLs, so large screenshots or
    photos grow by roughly a third before they reach the OpenAI-compatible API.
    Compress only oversized image payloads and preserve small images as-is.
    """
    max_bytes = _env_int("AGENT_VISION_MAX_INPUT_IMAGE_BYTES", _DEFAULT_MAX_VISION_IMAGE_BYTES, minimum=256 * 1024)
    if len(image_bytes or b"") <= max_bytes:
        return image_bytes, content_type
    try:
        from PIL import Image as PILImage, ImageOps
    except ImportError:
        return image_bytes, content_type

    max_dimension = _env_int("AGENT_VISION_MAX_INPUT_IMAGE_DIMENSION", _DEFAULT_MAX_VISION_IMAGE_DIMENSION, minimum=128)
    quality = _env_int("AGENT_VISION_JPEG_QUALITY", _DEFAULT_VISION_IMAGE_QUALITY, minimum=35, maximum=95)
    try:
        with PILImage.open(io.BytesIO(image_bytes)) as opened:
            source = ImageOps.exif_transpose(opened)
            if source.mode in {"RGBA", "LA"} or (source.mode == "P" and "transparency" in source.info):
                rgba = source.convert("RGBA")
                background = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                source = background.convert("RGB")
            elif source.mode != "RGB":
                source = source.convert("RGB")

            best: bytes | None = None
            dimension = max_dimension
            while True:
                for current_quality in _quality_steps(quality):
                    candidate = source.copy()
                    candidate.thumbnail((dimension, dimension), PILImage.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    candidate.save(buffer, format="JPEG", quality=current_quality, optimize=True, progressive=True)
                    data = buffer.getvalue()
                    if best is None or len(data) < len(best):
                        best = data
                    if len(data) <= max_bytes:
                        return data, "image/jpeg"
                if dimension <= 256:
                    break
                dimension = max(256, int(dimension * 0.7))
    except Exception:
        return image_bytes, content_type

    if not best:
        return image_bytes, content_type
    if len(best) < len(image_bytes) or len(image_bytes) > max_bytes:
        return best, "image/jpeg"
    return image_bytes, content_type


def _quality_steps(initial_quality: int) -> list[int]:
    steps = [initial_quality, 75, 65, 55, 45, 35]
    deduped: list[int] = []
    for value in steps:
        value = max(35, min(95, int(value)))
        if value not in deduped:
            deduped.append(value)
    return deduped


def _env_int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value
