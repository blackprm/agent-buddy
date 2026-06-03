from __future__ import annotations

import base64
import mimetypes
from typing import Any
from urllib.parse import urlparse

from agent_core.videos.base import GeneratedVideo, VideoGenerationResult, VideoInput


_SUCCESS_STATUSES = {"succeeded", "success", "done", "completed", "complete"}
_RUNNING_STATUSES = {"queued", "running", "processing", "pending", ""}
_FAILED_STATUSES = {"failed", "fail", "error", "cancelled", "canceled", "timeout"}


class BearerVideoGenerationClient:
    """Bearer-token video generation client for the AI video creative API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 30.0,
        poll_interval: float = 10.0,
        max_polls: int = 30,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError("video generation requires the 'httpx' package") from exc
        if not base_url:
            raise ValueError("video generation requires base_url")
        if not token:
            raise ValueError("video generation requires bearer token")
        self._httpx = httpx
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._max_polls = max(1, int(max_polls or 1))

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
        meta = dict(metadata or {})
        payload: dict[str, Any] = {
            "prompt": str(prompt or "").strip(),
            "mode": _enum(str(meta.get("mode") or mode or "fast"), {"fast", "pro"}, "fast"),
            "ratio": _enum(str(meta.get("ratio") or ratio or "adaptive"), {"16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"}, "adaptive"),
            "duration": int(meta.get("duration") or duration or 5),
            "resolution": _enum(str(meta.get("resolution") or resolution or "720p"), {"480p", "720p"}, "720p"),
            "generate_audio": bool(meta.get("generate_audio", generate_audio)),
            "watermark": bool(meta.get("watermark", watermark)),
        }
        for key in ("pe_mode", "rewrite_thinking_level", "seed", "web_search", "auto_queued"):
            if key in meta and meta[key] not in (None, ""):
                payload[key] = meta[key]
        payload.update(_media_payload("images", images or []))
        payload.update(_media_payload("videos", videos or []))
        payload.update(_media_payload("audios", audios or []))
        extra = meta.get("video_payload") or meta.get("payload")
        if isinstance(extra, dict):
            payload.update(extra)

        create_payload = await self._request("POST", "/create", json=payload)
        task_id = str(create_payload.get("task_id") or create_payload.get("taskId") or "").strip()
        if not task_id:
            video_urls = _video_urls(create_payload)
            if video_urls:
                videos = [await self._download_video(url, task_id="") for url in video_urls]
                return VideoGenerationResult(videos=videos, metadata={"provider": "bearer_video", "requested": payload, "raw_response": create_payload})
            raise ValueError("video create response did not contain task_id or video_url")

        last_payload: Any = create_payload
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_polls + 1):
                if attempt > 1 and self._poll_interval > 0:
                    await _sleep(self._poll_interval)
                last_payload = await self._request("GET", f"/status/{task_id}", client=client)
                status = str(last_payload.get("status") or "").strip().lower()
                if status in _SUCCESS_STATUSES:
                    urls = _video_urls(last_payload)
                    if not urls:
                        raise ValueError("video task succeeded but status response did not contain video_url")
                    videos = [await self._download_video(url, task_id=task_id, client=client, metadata=last_payload) for url in urls]
                    return VideoGenerationResult(videos=videos, metadata={"provider": "bearer_video", "task_id": task_id, "requested": payload, "raw_response": last_payload})
                if status in _FAILED_STATUSES:
                    message = last_payload.get("message") or last_payload.get("error") or last_payload.get("code") or status
                    raise ValueError(f"video generation task failed: {message}")
                if status not in _RUNNING_STATUSES:
                    urls = _video_urls(last_payload)
                    if urls:
                        videos = [await self._download_video(url, task_id=task_id, client=client, metadata=last_payload) for url in urls]
                        return VideoGenerationResult(videos=videos, metadata={"provider": "bearer_video", "task_id": task_id, "requested": payload, "raw_response": last_payload})
        raise TimeoutError(f"video generation task timed out after {self._max_polls} polls: {task_id}; last_response={last_payload}")

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        return await self._request("DELETE", f"/cancel/{task_id}")

    async def list_tasks(self, *, status: str = "", limit: int = 20, offset: int = 0) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": max(1, min(int(limit or 20), 100)), "offset": max(0, int(offset or 0))}
        if status:
            params["status"] = status
        return await self._request("GET", "/tasks", params=params)

    async def usage(self, *, start_date: str = "", end_date: str = "", mode: str = "") -> dict[str, Any]:
        params = {k: v for k, v in {"start_date": start_date, "end_date": end_date, "mode": mode}.items() if v}
        return await self._request("GET", "/usage", params=params)

    async def rewrite_prompt(
        self,
        *,
        prompt: str,
        pe_mode: str = "creative",
        duration: int = 8,
        rewrite_thinking_level: str = "standard",
        images: list[VideoInput] | None = None,
        videos: list[VideoInput] | None = None,
        audios: list[VideoInput] | None = None,
        image_url: str = "",
        wait: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": str(prompt or "").strip(),
            "pe_mode": _enum(str(pe_mode or "creative"), {"auto", "creative", "reference"}, "creative"),
            "duration": int(duration or 8),
            "rewrite_thinking_level": _enum(str(rewrite_thinking_level or "standard"), {"standard", "accelerated"}, "standard"),
        }
        if image_url:
            payload["image_url"] = image_url
        payload.update(_media_payload("images", images or []))
        payload.update(_media_payload("videos", videos or []))
        payload.update(_media_payload("audios", audios or []))
        created = await self._request("POST", "/prompt-rewrite", json=payload)
        task_id = str(created.get("task_id") or "").strip()
        if not wait or not task_id:
            return created
        last_payload: dict[str, Any] = created
        async with self._httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._max_polls + 1):
                if attempt > 1:
                    await _sleep(max(1.0, min(self._poll_interval, 5.0)))
                last_payload = await self._request("GET", f"/prompt-rewrite/{task_id}", client=client)
                status = str(last_payload.get("status") or "").strip().lower()
                if status == "completed":
                    return last_payload
                if status in {"failed", "timeout", "cancelled", "canceled"}:
                    message = last_payload.get("message") or last_payload.get("error") or status
                    raise ValueError(f"prompt rewrite task failed: {message}")
        raise TimeoutError(f"prompt rewrite task timed out after {self._max_polls} polls: {task_id}; last_response={last_payload}")

    async def cancel_rewrite(self, task_id: str) -> dict[str, Any]:
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        return await self._request("DELETE", f"/prompt-rewrite/{task_id}")

    async def create_asset_group(self, *, name: str, description: str = "") -> dict[str, Any]:
        payload = {"name": name}
        if description:
            payload["description"] = description
        return await self._request("POST", "/assets/groups", json=payload)

    async def list_asset_groups(self, **params: Any) -> dict[str, Any]:
        return await self._request("GET", "/assets/groups", params=_clean_params(params))

    async def get_asset_group(self, group_id: str) -> dict[str, Any]:
        group_id = str(group_id or "").strip()
        if not group_id:
            raise ValueError("group_id is required")
        return await self._request("GET", f"/assets/groups/{group_id}")

    async def update_asset_group(self, group_id: str, *, name: str = "", description: str = "") -> dict[str, Any]:
        group_id = str(group_id or "").strip()
        if not group_id:
            raise ValueError("group_id is required")
        payload = _clean_params({"name": name, "description": description})
        if not payload:
            raise ValueError("name or description is required")
        return await self._request("PUT", f"/assets/groups/{group_id}", json=payload)

    async def upload_asset(self, *, group_id: str, url: str, name: str = "", asset_type: str = "Image") -> dict[str, Any]:
        payload = {"group_id": group_id, "url": url, "asset_type": asset_type or "Image"}
        if name:
            payload["name"] = name
        return await self._request("POST", "/assets/upload", json=payload)

    async def list_assets(self, **params: Any) -> dict[str, Any]:
        return await self._request("GET", "/assets", params=_clean_params(params))

    async def get_asset(self, asset_id: str) -> dict[str, Any]:
        asset_id = str(asset_id or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        return await self._request("GET", f"/assets/{asset_id}")

    async def update_asset(self, asset_id: str, *, name: str) -> dict[str, Any]:
        asset_id = str(asset_id or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        if not name:
            raise ValueError("name is required")
        return await self._request("PUT", f"/assets/{asset_id}", json={"name": name})

    async def delete_asset(self, asset_id: str) -> dict[str, Any]:
        asset_id = str(asset_id or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        return await self._request("DELETE", f"/assets/{asset_id}")

    async def create_validate_session(self, *, callback_url: str) -> dict[str, Any]:
        if not callback_url:
            raise ValueError("callback_url is required")
        return await self._request("POST", "/assets/validate-session", json={"callback_url": callback_url})

    async def get_validate_result(self, *, byted_token: str) -> dict[str, Any]:
        if not byted_token:
            raise ValueError("byted_token is required")
        return await self._request("POST", "/assets/validate-result", json={"byted_token": byted_token})

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None, client: Any | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if json is not None:
            headers["Content-Type"] = "application/json"
        url = f"{self._base_url}{path}"
        if client is not None:
            response = await client.request(method, url, headers=headers, json=json, params=params)
        else:
            async with self._httpx.AsyncClient(timeout=self._timeout) as scoped_client:
                response = await scoped_client.request(method, url, headers=headers, json=json, params=params)
        return _json_or_error(response)

    async def _download_video(self, url: str, *, task_id: str, client: Any | None = None, metadata: dict[str, Any] | None = None) -> GeneratedVideo:
        if client is not None:
            response = await client.get(url)
        else:
            async with self._httpx.AsyncClient(timeout=self._timeout) as scoped_client:
                response = await scoped_client.get(url)
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError(f"failed to download generated video: HTTP {response.status_code}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if not content_type:
            guessed, _ = mimetypes.guess_type(urlparse(url).path)
            content_type = guessed or "video/mp4"
        return GeneratedVideo(data=response.content, content_type=content_type, raw_url=url, task_id=task_id, metadata=dict(metadata or {}))


def _media_payload(field: str, items: list[VideoInput]) -> dict[str, Any]:
    if not items:
        return {}
    return {field: [{"url": item.url, "role": item.role} for item in items if item.url]}


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value not in (None, "", [])}


def _enum(value: str, allowed: set[str], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _video_url(payload: dict[str, Any]) -> str:
    urls = _video_urls(payload)
    return urls[0] if urls else ""


def _video_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("video_url", "videoUrl", "url", "result_url", "resultUrl"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            urls.append(value)
    for key in ("video_urls", "videoUrls", "urls", "result_urls", "resultUrls"):
        value = payload.get(key)
        if isinstance(value, list):
            urls.extend(str(item) for item in value if isinstance(item, str) and item.startswith(("http://", "https://")))
    for key in ("videos", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.startswith(("http://", "https://")):
                    urls.append(item)
                elif isinstance(item, dict):
                    urls.extend(_video_urls(item))
    data = payload.get("data")
    if isinstance(data, dict):
        urls.extend(_video_urls(data))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _json_or_error(response: Any) -> dict[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        try:
            payload = response.json()
            detail = payload.get("message") or payload.get("error") or payload.get("detail") or payload
        except Exception:
            detail = getattr(response, "text", "")
        raise ValueError(f"video API request failed ({response.status_code}): {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("video API response was not a JSON object")
    return payload


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def image_bytes_to_data_url(data: bytes, content_type: str) -> str:
    return f"data:{content_type or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"
