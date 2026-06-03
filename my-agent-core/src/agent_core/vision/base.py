from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VisionResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisionMediaInput:
    data: bytes
    content_type: str
    filename: str = ""
    attachment_id: str = ""


class VisionClient(Protocol):
    async def understand_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        ...

    async def understand_video(
        self,
        *,
        video_bytes: bytes,
        content_type: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        ...

    async def understand_media(
        self,
        *,
        media: list[VisionMediaInput],
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> VisionResult:
        ...
