from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class VideoInput:
    url: str
    role: str


@dataclass(slots=True)
class GeneratedVideo:
    data: bytes
    content_type: str = "video/mp4"
    raw_url: str = ""
    task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoGenerationResult:
    videos: list[GeneratedVideo]
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoGenerationClient(Protocol):
    async def generate_video(
        self,
        *,
        prompt: str,
        mode: str = "fast",
        ratio: str = "adaptive",
        duration: int = 5,
        resolution: str = "720p",
        generate_audio: bool = True,
        watermark: bool = False,
        images: list[VideoInput] | None = None,
        videos: list[VideoInput] | None = None,
        audios: list[VideoInput] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoGenerationResult:
        ...
