from __future__ import annotations

import base64
import json
from typing import Any

from agent_core.images.base import GeneratedImage, ImageGenerationResult, ImageInput


CODEX_PROMPT_PREFIX = "Use the following text as the complete prompt. Do not rewrite it:\n\n"
_FORMAT_TO_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp"}


class OpenAICompatibleImageGenerationClient:
    """OpenAI Images/Responses compatible image generation and editing client.

    This intentionally ports the core compatibility model from gpt_image_playground:
    Codex mode prefixes prompts to prevent rewriting, omits unsupported quality
    parameters, supports Images API generate/edit, and can use Responses API with
    the image_generation tool.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_mode: str = "images",
        codex_cli: bool = False,
        timeout: float = 120.0,
        image_field: str = "image[]",
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError("image generation requires the 'httpx' package") from exc

        self._httpx = httpx
        self._model = model
        self._api_key = api_key or ""
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_mode = "responses" if api_mode == "responses" else "images"
        self._codex_cli = codex_cli
        self._timeout = timeout
        self._image_field = image_field or "image[]"

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
        params = self._params(size=size, quality=quality, output_format=output_format, n=n, metadata=metadata)
        if self._api_mode == "responses":
            return await self._responses_image(prompt=prompt, input_images=[], mask=None, **params)
        return await self._images_generate(prompt=prompt, **params)

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
        if not input_images:
            raise ValueError("edit_image requires at least one input image")
        params = self._params(size=size, quality=quality, output_format=output_format, n=n, metadata=metadata)
        if self._api_mode == "responses":
            return await self._responses_image(prompt=prompt, input_images=input_images, mask=mask, **params)
        return await self._images_edit(prompt=prompt, input_images=input_images, mask=mask, **params)

    def _params(self, *, size: str, quality: str | None, output_format: str, n: int, metadata: dict[str, Any] | None) -> dict[str, Any]:
        meta = dict(metadata or {})
        fmt = _normalize_format(str(meta.get("output_format") or output_format or "png"))
        count = max(1, min(int(meta.get("n") or n or 1), int(meta.get("max_n") or 10)))
        normalized_quality = str(meta.get("quality") or quality or "").strip() or None
        if self._codex_cli:
            normalized_quality = None
        return {
            "size": _normalize_size(str(meta.get("size") or size or "1024x1024")),
            "quality": normalized_quality,
            "output_format": fmt,
            "n": count,
            "moderation": str(meta.get("moderation") or "auto"),
            "output_compression": meta.get("output_compression"),
        }

    def _prompt(self, prompt: str) -> str:
        text = str(prompt or "").strip()
        return CODEX_PROMPT_PREFIX + text if self._codex_cli else text

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _images_generate(self, *, prompt: str, size: str, quality: str | None, output_format: str, n: int, moderation: str, output_compression: Any) -> ImageGenerationResult:
        body: dict[str, Any] = {
            "model": self._model,
            "prompt": self._prompt(prompt),
            "size": size,
            "output_format": output_format,
            "moderation": moderation,
            "response_format": "b64_json",
        }
        if quality:
            body["quality"] = quality
        if n > 1:
            body["n"] = n
        if output_format != "png" and output_compression not in (None, ""):
            body["output_compression"] = output_compression
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/images/generations", headers={**self._headers(), "Content-Type": "application/json"}, json=body)
        payload = _json_or_error(response)
        return await self._parse_result(payload, requested={"api_mode": "images", "operation": "generate", **body})

    async def _images_edit(self, *, prompt: str, input_images: list[ImageInput], mask: ImageInput | None, size: str, quality: str | None, output_format: str, n: int, moderation: str, output_compression: Any) -> ImageGenerationResult:
        data: dict[str, str] = {
            "model": self._model,
            "prompt": self._prompt(prompt),
            "size": size,
            "output_format": output_format,
            "moderation": moderation,
            "response_format": "b64_json",
        }
        if quality:
            data["quality"] = quality
        if n > 1:
            data["n"] = str(n)
        if output_format != "png" and output_compression not in (None, ""):
            data["output_compression"] = str(output_compression)
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for idx, image in enumerate(input_images, 1):
            files.append((self._image_field, (image.filename or f"input-{idx}.png", image.data, image.content_type or "image/png")))
        if mask is not None:
            files.append(("mask", (mask.filename or "mask.png", mask.data, mask.content_type or "image/png")))
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/images/edits", headers=self._headers(), data=data, files=files)
        payload = _json_or_error(response)
        return await self._parse_result(payload, requested={"api_mode": "images", "operation": "edit", **data, "input_images": len(input_images), "has_mask": mask is not None})

    async def _responses_image(self, *, prompt: str, input_images: list[ImageInput], mask: ImageInput | None, size: str, quality: str | None, output_format: str, n: int, moderation: str, output_compression: Any) -> ImageGenerationResult:
        images = [_data_url(item.data, item.content_type) for item in input_images]
        text = self._prompt(prompt)
        if images:
            content = [{"type": "input_text", "text": text}] + [{"type": "input_image", "image_url": url} for url in images]
            input_payload: Any = [{"role": "user", "content": content}]
        else:
            input_payload = text
        tool: dict[str, Any] = {
            "type": "image_generation",
            "action": "edit" if images else "generate",
            "size": size,
            "output_format": output_format,
            "moderation": moderation,
        }
        if quality:
            tool["quality"] = quality
        if output_format != "png" and output_compression not in (None, ""):
            tool["output_compression"] = output_compression
        if mask is not None:
            tool["input_image_mask"] = {"image_url": _data_url(mask.data, mask.content_type)}
        body = {"model": self._model, "input": input_payload, "tools": [tool], "tool_choice": "required"}
        requests = max(1, n)
        payloads: list[dict[str, Any]] = []
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for _ in range(requests):
                response = await client.post(f"{self._base_url}/responses", headers={**self._headers(), "Content-Type": "application/json"}, json=body)
                payloads.append(_json_or_error(response))
        result = await self._parse_result(payloads, requested={"api_mode": "responses", "operation": tool["action"], **tool, "n": n})
        return result

    async def _parse_result(self, payload: Any, *, requested: dict[str, Any]) -> ImageGenerationResult:
        candidates: list[dict[str, Any]] = []
        for item in _walk_image_candidates(payload):
            candidates.append(item)
        images: list[GeneratedImage] = []
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for item in candidates:
                raw = item.get("b64_json") or item.get("base64") or item.get("image") or item.get("data") or item.get("result") or item.get("url")
                if not isinstance(raw, str) or not raw:
                    continue
                data, content_type, raw_url = await _image_bytes_from_value(raw, client, _FORMAT_TO_MIME.get(str(requested.get("output_format") or "png"), "image/png"))
                images.append(GeneratedImage(data=data, content_type=content_type, revised_prompt=str(item.get("revised_prompt") or item.get("revisedPrompt") or ""), raw_url=raw_url, metadata={"raw_item_keys": sorted(item.keys())}))
        if not images:
            raise ValueError("image API response did not contain image data")
        return ImageGenerationResult(images=images, metadata={"model": self._model, "codex_cli": self._codex_cli, "requested": requested, "raw_response": payload})


def _normalize_format(value: str) -> str:
    value = value.lower().strip().lstrip(".")
    return "jpeg" if value == "jpg" else (value if value in {"png", "jpeg", "webp"} else "png")


def _normalize_size(value: str) -> str:
    value = str(value or "").strip().lower()
    aliases = {"1k": "1024x1024", "2k": "2048x2048", "4k": "4096x4096", "square": "1024x1024"}
    value = aliases.get(value, value)
    if value == "auto":
        return value
    if "x" not in value:
        return "1024x1024"
    left, right = value.split("x", 1)
    try:
        width = max(256, min(int(left), 4096))
        height = max(256, min(int(right), 4096))
    except ValueError:
        return "1024x1024"
    return f"{width}x{height}"


def _data_url(data: bytes, content_type: str) -> str:
    return f"data:{content_type or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"


def _json_or_error(response: Any) -> dict[str, Any]:
    text = response.text
    if response.status_code < 200 or response.status_code >= 300:
        try:
            payload = response.json()
            detail = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("detail") or payload.get("message")
        except Exception:
            detail = text
        raise ValueError(f"image API request failed ({response.status_code}): {detail or response.reason_phrase}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("image API response was not valid JSON") from exc


def _walk_image_candidates(payload: Any):
    if isinstance(payload, list):
        for item in payload:
            yield from _walk_image_candidates(item)
        return
    if not isinstance(payload, dict):
        return
    if any(isinstance(payload.get(key), str) for key in ("b64_json", "base64", "image", "data", "result", "url")):
        yield payload
    for key in ("data", "output", "images", "image", "results", "tools"):
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            yield from _walk_image_candidates(value)


async def _image_bytes_from_value(value: str, client: Any, fallback_content_type: str) -> tuple[bytes, str, str]:
    if value.startswith("data:"):
        header, encoded = value.split(",", 1)
        content_type = header[5:].split(";", 1)[0] or fallback_content_type
        return base64.b64decode(encoded), content_type, ""
    if value.startswith("http://") or value.startswith("https://"):
        response = await client.get(value)
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError(f"failed to download generated image: HTTP {response.status_code}")
        return response.content, response.headers.get("content-type", "").split(";", 1)[0] or fallback_content_type, value
    return base64.b64decode(value), fallback_content_type, ""
