from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ImageInput:
    data: bytes
    content_type: str
    filename: str = "image.png"


@dataclass(slots=True)
class GeneratedImage:
    data: bytes
    content_type: str = "image/png"
    revised_prompt: str = ""
    raw_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageGenerationResult:
    images: list[GeneratedImage]
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageGenerationClient(Protocol):
    async def generate_image(
        self,
        *,
        prompt: str,
        size: str = "1024x1024",
        quality: str | None = None,
        output_format: str = "png",
        n: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        ...

    async def edit_image(
        self,
        *,
        prompt: str,
        input_images: list[ImageInput],
        mask: ImageInput | None = None,
        size: str = "1024x1024",
        quality: str | None = None,
        output_format: str = "png",
        n: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> ImageGenerationResult:
        ...
