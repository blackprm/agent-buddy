from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, quote

from agent_core.images.base import GeneratedImage, ImageGenerationResult, ImageInput


_FORMAT_TO_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg", "webp": "image/webp"}
_SUCCESS_STATUSES = {"done", "success", "succeeded", "finished", "completed", "complete"}
_RUNNING_STATUSES = {"", "pending", "queued", "created", "running", "processing", "submitted", "in_progress"}
_FAILED_STATUSES = {"failed", "fail", "error", "canceled", "cancelled", "timeout", "expired"}
_DEFAULT_MAX_INPUT_IMAGE_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_INPUT_IMAGE_DIMENSION = 2048
_DEFAULT_INPUT_IMAGE_QUALITY = 85


class AIDPImageGenerationClient:
    """Volcengine AIDP image generation client via VolSign (AWS SigV4-style) signing.

    Talks to open.volcengineapi.com with CDP SaaS service credentials.
    Payload uses token/kind/prompt/size/quality/n fields matching the AIDP gateway.
    """

    def __init__(
        self,
        *,
        access_key_id: str,
        secret_access_key: str,
        token: str,
        host: str = "open.volcengineapi.com",
        service: str = "cdp_saas",
        region: str = "cn-beijing",
        version: str = "2022-08-01",
        timeout: float = 120.0,
        poll_interval: float = 6.0,
        max_polls: int = 60,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("AIDP image generation requires the 'httpx' package") from exc

        if not access_key_id:
            raise ValueError("AIDP image generation requires access_key_id")
        if not secret_access_key:
            raise ValueError("AIDP image generation requires secret_access_key")
        if not token:
            raise ValueError("AIDP image generation requires token")

        self._httpx = httpx
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._token = token
        self._host = host
        self._service = service
        self._region = region
        self._version = version
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_polls = max(1, int(max_polls or 1))

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
        payload = self._create_payload(
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            input_images=[],
            metadata=metadata,
        )
        return await self._create_and_poll(payload=payload)

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
        payload = self._create_payload(
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            input_images=input_images,
            metadata=metadata,
        )
        return await self._create_and_poll(payload=payload)

    def _create_payload(
        self,
        *,
        prompt: str,
        size: str,
        quality: str | None,
        n: int,
        input_images: list[ImageInput],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        meta = dict(metadata or {})
        payload: dict[str, Any] = {
            "token": self._token,
            "kind": "edit" if input_images else "generation",
            "size": str(meta.get("size") or size or "1024x1024"),
            "n": max(1, min(int(meta.get("n") or n or 1), int(meta.get("max_n") or 10))),
        }
        if prompt:
            payload["prompt"] = str(prompt or "").strip()
        normalized_quality = str(meta.get("quality") or quality or "").strip()
        if normalized_quality:
            payload["quality"] = normalized_quality
        if input_images:
            prepared_images = [_compress_input_image_if_needed(img) for img in input_images]
            payload["images"] = [
                {
                    "b64": base64.b64encode(img.data).decode("ascii"),
                    "filename": img.filename or "input.png",
                    "content_type": img.content_type or "image/png",
                }
                for img in prepared_images
            ]
        return payload

    async def _create_and_poll(self, *, payload: dict[str, Any]) -> ImageGenerationResult:
        create_payload = await self._signed_post(action="CreateImageTask", payload=payload)
        task_id = _find_first_string(create_payload, ("task_id", "taskId", "TaskId", "taskID"))
        if not task_id:
            raise ValueError("AIDP CreateImageTask response did not contain task_id")

        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_polls + 1):
                if attempt > 1 and self._poll_interval > 0:
                    await _sleep(self._poll_interval)
                status_payload = {"task_id": task_id, "token": self._token}
                last_payload = await self._signed_post(action="GetImageTaskStatus", payload=status_payload, client=client)
                status = _find_first_string(last_payload, ("status", "Status", "task_status", "taskStatus", "state", "State")).lower()
                if status in _SUCCESS_STATUSES:
                    images = await self._extract_images(last_payload, client=client)
                    if images:
                        return ImageGenerationResult(
                            images=images,
                            metadata={"provider": "aidp_volsign", "task_id": task_id, "requested": payload, "raw_response": last_payload},
                        )
                    raise ValueError("AIDP task succeeded but response did not contain image data")
                if status in _FAILED_STATUSES:
                    message = _find_first_string(last_payload, ("message", "Message", "msg", "Msg", "error", "Error", "reason", "Reason"))
                    raise ValueError(f"AIDP image task failed: {message or status}")
        raise TimeoutError(f"AIDP image task timed out after {self._max_polls} polls: {task_id}")

    async def _signed_post(self, *, action: str, payload: dict[str, Any], client: Any | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        query = {"Action": action, "Version": self._version}
        headers = _build_volsign_headers(
            method="POST",
            path="/",
            query=query,
            body=body,
            host=self._host,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            service=self._service,
            region=self._region,
        )
        url = f"https://{self._host}/"
        if client is not None:
            response = await client.post(url, params=query, content=body.encode("utf-8"), headers=headers)
        else:
            async with self._httpx.AsyncClient(timeout=self._timeout) as scoped_client:
                response = await scoped_client.post(url, params=query, content=body.encode("utf-8"), headers=headers)
        return _json_or_error(response)

    async def _extract_images(self, payload: Any, *, client: Any) -> list[GeneratedImage]:
        urls = _extract_urls(payload)
        if not urls:
            return []
        images: list[GeneratedImage] = []
        for url in urls:
            data, content_type = await _download_image(url, client)
            images.append(GeneratedImage(data=data, content_type=content_type, raw_url=url, metadata={"provider": "aidp_volsign"}))
        return images


class TopAidpImageGenerationClient:
    """Backward-compatible TOP/AIDP client facade.

    Older server code and tests import ``TopAidpImageGenerationClient`` and pass
    TOP-style constructor arguments.  The production Desktop-backed path now
    uses VolSign/CDP SaaS, but we keep this facade so startup and existing
    integrations do not break.
    """

    def __init__(
        self,
        *,
        app_key: str,
        app_secret: str,
        base_url: str,
        model: str = "",
        api_version: str = "2.0",
        create_method: str = "CreateImageTask",
        status_method: str = "GetImageTaskStatus",
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        max_polls: int = 90,
        output_url: bool = True,
        signature_protocol: str = "top",
        volc_service: str = "cdp_saas",
        volc_region: str = "cn-beijing",
        aidp_token: str = "",
        session_token: str = "",
    ) -> None:
        del session_token  # Volc temporary session token is not used by this gateway.
        self._signature_protocol = (signature_protocol or "top").strip().lower()
        self._model = model or ""
        if self._signature_protocol in {"volsign", "volc", "volcengine", "sigv4"}:
            parsed = urlparse(base_url or "https://open.volcengineapi.com")
            host = parsed.netloc or parsed.path or "open.volcengineapi.com"
            token = aidp_token or model
            self._delegate = AIDPImageGenerationClient(
                access_key_id=app_key,
                secret_access_key=app_secret,
                token=token,
                host=host,
                service=volc_service or "cdp_saas",
                region=volc_region or "cn-beijing",
                version=api_version or "2022-08-01",
                timeout=timeout,
                poll_interval=poll_interval,
                max_polls=max_polls,
            )
            return

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError("Top AIDP image generation requires the 'httpx' package") from exc
        if not app_key:
            raise ValueError("Top AIDP image generation requires app_key")
        if not app_secret:
            raise ValueError("Top AIDP image generation requires app_secret")
        if not base_url:
            raise ValueError("Top AIDP image generation requires base_url")
        self._delegate = None
        self._httpx = httpx
        self._app_key = app_key
        self._app_secret = app_secret
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version or "2.0"
        self._create_method = create_method or "CreateImageTask"
        self._status_method = status_method or "GetImageTaskStatus"
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_polls = max(1, int(max_polls or 1))
        self._output_url = output_url

    async def generate_image(self, **kwargs: Any) -> ImageGenerationResult:
        if self._delegate is not None:
            return await self._delegate.generate_image(**kwargs)
        payload = self._create_top_payload(input_images=[], **kwargs)
        return await self._create_and_poll_top(payload=payload, requested={"operation": "generate", **payload})

    async def edit_image(self, **kwargs: Any) -> ImageGenerationResult:
        if self._delegate is not None:
            return await self._delegate.edit_image(**kwargs)
        input_images = list(kwargs.get("input_images") or [])
        if not input_images:
            raise ValueError("edit_image requires at least one input image")
        payload = self._create_top_payload(**{**kwargs, "input_images": input_images})
        return await self._create_and_poll_top(payload=payload, requested={"operation": "edit", **payload})

    def _create_top_payload(
        self,
        *,
        prompt: str,
        size: str = "1024x1024",
        quality: str | None = None,
        output_format: str = "png",
        n: int = 1,
        metadata: dict[str, Any] | None = None,
        input_images: list[ImageInput],
        mask: ImageInput | None = None,
    ) -> dict[str, Any]:
        del mask
        meta = dict(metadata or {})
        width, height = _parse_size(str(meta.get("size") or size or "1024x1024"))
        fmt = _normalize_format(str(meta.get("output_format") or output_format or "png"))
        payload: dict[str, Any] = {
            "prompt": str(prompt or "").strip(),
            "size": f"{width}x{height}",
            "width": width,
            "height": height,
            "n": max(1, min(int(meta.get("n") or n or 1), int(meta.get("max_n") or 10))),
            "output_format": fmt,
            "return_url": bool(meta.get("return_url", self._output_url)),
        }
        if self._model:
            payload["model"] = self._model
            payload["req_key"] = self._model
        normalized_quality = str(meta.get("quality") or quality or "").strip()
        if normalized_quality:
            payload["quality"] = normalized_quality
        if input_images:
            prepared_images = [_compress_input_image_if_needed(image) for image in input_images]
            payload["image_base64_list"] = [base64.b64encode(image.data).decode("ascii") for image in prepared_images]
            payload["image_mime_types"] = [image.content_type or "image/png" for image in prepared_images]
        return payload

    async def _create_and_poll_top(self, *, payload: dict[str, Any], requested: dict[str, Any]) -> ImageGenerationResult:
        create_payload = await self._post_top(method=self._create_method, payload=payload)
        task_id = _find_first_string(create_payload, ("task_id", "taskId", "TaskId", "taskID", "id", "Id"))
        if not task_id:
            raise ValueError("Top AIDP CreateImageTask response did not contain task_id")
        status_payload: dict[str, Any] = {"task_id": task_id, "taskId": task_id}
        if self._model:
            status_payload["model"] = self._model
            status_payload["req_key"] = self._model
        last_payload: Any = create_payload
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_polls + 1):
                if attempt > 1 and self._poll_interval > 0:
                    await _sleep(self._poll_interval)
                last_payload = await self._post_top(method=self._status_method, payload=status_payload, client=client)
                status = _find_first_string(last_payload, ("status", "Status", "task_status", "taskStatus", "state", "State")).lower()
                if status in _SUCCESS_STATUSES:
                    urls = _extract_urls(last_payload)
                    images = []
                    for url in urls:
                        data, content_type = await _download_image(url, client)
                        images.append(GeneratedImage(data=data, content_type=content_type, raw_url=url, metadata={"provider": "top_aidp"}))
                    if images:
                        return ImageGenerationResult(images=images, metadata={"provider": "top_aidp", "model": self._model, "task_id": task_id, "requested": requested, "raw_response": last_payload})
                    raise ValueError("Top AIDP task succeeded but response did not contain image data")
                if status in _FAILED_STATUSES:
                    message = _find_first_string(last_payload, ("message", "Message", "msg", "Msg", "error", "Error", "reason", "Reason"))
                    raise ValueError(f"Top AIDP image task failed: {message or status}")
        raise TimeoutError(f"Top AIDP image task timed out after {self._max_polls} polls: {task_id}; last_response={last_payload}")

    async def _post_top(self, *, method: str, payload: dict[str, Any], client: Any | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        params = build_top_common_params(app_key=self._app_key, method=method, api_version=self._api_version)
        params["sign"] = sign_top_request(params=params, body=body, app_secret=self._app_secret)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if client is not None:
            response = await client.post(self._base_url, params=params, content=body.encode("utf-8"), headers=headers)
        else:
            async with self._httpx.AsyncClient(timeout=self._timeout) as scoped_client:
                response = await scoped_client.post(self._base_url, params=params, content=body.encode("utf-8"), headers=headers)
        return _json_or_error(response)


def _compress_input_image_if_needed(image: ImageInput) -> ImageInput:
    """Keep edit-reference image payloads under gateway limits before base64 encoding.

    AIDP edit requests inline input images as base64 JSON fields.  A raw 15MB
    upload becomes roughly 20MB after base64 and can be rejected by the gateway
    with HTTP 400 before the task is even created.  Compress oversized inputs to
    JPEG proactively while preserving smaller images as-is.
    """
    max_bytes = _env_int("AGENT_IMAGE_EDIT_MAX_INPUT_BYTES", _DEFAULT_MAX_INPUT_IMAGE_BYTES, minimum=256 * 1024)
    if len(image.data or b"") <= max_bytes:
        return image
    try:
        from PIL import Image as PILImage, ImageOps
    except ImportError:
        return image

    max_dimension = _env_int("AGENT_IMAGE_EDIT_MAX_INPUT_DIMENSION", _DEFAULT_MAX_INPUT_IMAGE_DIMENSION, minimum=256)
    quality = _env_int("AGENT_IMAGE_EDIT_JPEG_QUALITY", _DEFAULT_INPUT_IMAGE_QUALITY, minimum=35, maximum=95)
    try:
        with PILImage.open(io.BytesIO(image.data)) as opened:
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
            current_quality = quality
            while True:
                candidate = source.copy()
                candidate.thumbnail((dimension, dimension), PILImage.Resampling.LANCZOS)
                buffer = io.BytesIO()
                candidate.save(buffer, format="JPEG", quality=current_quality, optimize=True, progressive=True)
                data = buffer.getvalue()
                if best is None or len(data) < len(best):
                    best = data
                if len(data) <= max_bytes:
                    break
                if current_quality > 55:
                    current_quality = max(55, current_quality - 10)
                    continue
                if dimension > 768:
                    dimension = max(768, int(dimension * 0.75))
                    current_quality = quality
                    continue
                break
    except Exception:
        return image

    if not best or len(best) >= len(image.data):
        return image
    filename = Path(image.filename or "input.png").with_suffix(".jpg").name
    return ImageInput(data=best, content_type="image/jpeg", filename=filename)


def _env_int(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def build_top_common_params(*, app_key: str, method: str, api_version: str = "2.0", timestamp: str | None = None) -> dict[str, str]:
    return {
        "app_key": app_key,
        "method": method,
        "timestamp": timestamp or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "json",
        "v": api_version or "2.0",
        "sign_method": "hmac-sha256",
    }


def sign_top_request(*, params: dict[str, Any], body: str, app_secret: str) -> str:
    pieces: list[str] = []
    for key in sorted(params):
        if key == "sign":
            continue
        value = params.get(key)
        if key and value not in (None, ""):
            pieces.append(f"{key}{value}")
    if body:
        pieces.append(body)
    source = "".join(pieces)
    return hmac.new(app_secret.encode("utf-8"), source.encode("utf-8"), hashlib.sha256).hexdigest().upper()


def sign_volcengine_request(**kwargs: Any) -> dict[str, str]:
    return _build_volsign_headers(**kwargs)


def _build_volsign_headers(
    *,
    method: str,
    path: str,
    query: dict[str, str],
    body: str,
    host: str,
    access_key_id: str,
    secret_access_key: str,
    service: str,
    region: str,
    now: _dt.datetime | None = None,
) -> dict[str, str]:
    current = now or _dt.datetime.now(_dt.UTC)
    if current.tzinfo is not None:
        current = current.astimezone(_dt.UTC).replace(tzinfo=None)
    x_date = current.strftime("%Y%m%dT%H%M%SZ")
    short_date = x_date[:8]
    content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    content_type = "application/json" if body else "application/x-www-form-urlencoded"

    canonical_query = "&".join(
        f"{quote(str(k), safe='-_.~')}={quote(str(v), safe='-_.~')}"
        for k, v in sorted(query.items())
    )

    signed_headers = "host;x-date;x-content-sha256;content-type"
    canonical_request = "\n".join([
        method.upper(),
        path or "/",
        canonical_query,
        f"host:{host}",
        f"x-date:{x_date}",
        f"x-content-sha256:{content_sha256}",
        f"content-type:{content_type}",
        "",
        signed_headers,
        content_sha256,
    ])

    credential_scope = f"{short_date}/{region}/{service}/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        x_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    k_date = _hmac_sha256(secret_access_key.encode("utf-8"), short_date)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "Content-Type": content_type,
        "Accept": "application/json",
        "Host": host,
        "X-Date": x_date,
        "X-Content-Sha256": content_sha256,
        "Authorization": (
            f"HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }


def _hmac_sha256(key: bytes, content: str) -> bytes:
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


def _parse_size(value: str) -> tuple[int, int]:
    normalized = str(value or "").strip().lower()
    aliases = {"1k": "1024x1024", "2k": "2048x2048", "4k": "4096x4096", "square": "1024x1024", "auto": "1024x1024"}
    normalized = aliases.get(normalized, normalized)
    if "x" not in normalized:
        return 1024, 1024
    left, right = normalized.split("x", 1)
    try:
        return max(256, min(int(left), 4096)), max(256, min(int(right), 4096))
    except ValueError:
        return 1024, 1024


def _normalize_format(value: str) -> str:
    value = str(value or "").lower().strip().lstrip(".")
    return "jpeg" if value == "jpg" else (value if value in {"png", "jpeg", "webp"} else "png")


def _json_or_error(response: Any) -> dict[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        try:
            payload = response.json()
            detail = payload.get("message") or payload.get("msg") or payload.get("error") or payload.get("detail")
        except Exception:
            detail = getattr(response, "text", "")
        raise ValueError(f"AIDP request failed ({response.status_code}): {detail or getattr(response, 'reason_phrase', '')}")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("AIDP response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("AIDP response was not a JSON object")
    return payload


def _find_first_string(payload: Any, keys: tuple[str, ...]) -> str:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
        for value in payload.values():
            found = _find_first_string(value, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_string(item, keys)
            if found:
                return found
    return ""


def _extract_urls(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        for key in ("urls", "Urls", "result_urls", "resultUrls", "image_urls", "imageUrls"):
            value = payload.get(key)
            if isinstance(value, list):
                urls = [str(u) for u in value if isinstance(u, str) and u.startswith("http")]
                if urls:
                    return urls
        for value in payload.values():
            found = _extract_urls(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_urls(item)
            if found:
                return found
    return []


async def _download_image(url: str, client: Any) -> tuple[bytes, str]:
    response = await client.get(url)
    if response.status_code < 200 or response.status_code >= 300:
        raise ValueError(f"failed to download generated image: HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if not content_type:
        guessed, _ = mimetypes.guess_type(urlparse(url).path)
        content_type = guessed or "image/png"
    return response.content, content_type


async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
