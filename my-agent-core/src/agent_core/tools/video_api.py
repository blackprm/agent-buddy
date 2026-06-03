from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from agent_core.recovery.tool_errors import format_exception_detail
from agent_core.tools.base import ToolContext, ToolResult
from agent_core.videos import VideoGenerationClient, VideoInput


class VideoApiTool:
    name = "VideoApi"
    description = "Operate the configured AI video creative API beyond generation: prompt rewrite, cancel/list tasks, usage, asset groups, assets, and liveness validation sessions. Use GenerateVideo for actual video generation."
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "rewrite_prompt", "cancel_rewrite", "cancel_task", "list_tasks", "usage",
                    "create_asset_group", "list_asset_groups", "get_asset_group", "update_asset_group",
                    "upload_asset", "list_assets", "get_asset", "update_asset", "delete_asset",
                    "create_validate_session", "get_validate_result",
                ],
            },
            "prompt": {"type": "string"},
            "pe_mode": {"type": "string", "enum": ["auto", "creative", "reference"]},
            "rewrite_thinking_level": {"type": "string", "enum": ["standard", "accelerated"]},
            "duration": {"type": "integer"},
            "wait": {"type": "boolean", "description": "For rewrite_prompt, wait and poll until completed."},
            "task_id": {"type": "string"},
            "status": {"type": "string"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "mode": {"type": "string", "enum": ["", "fast", "pro"]},
            "group_id": {"type": "string"},
            "asset_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "url": {"type": "string", "description": "Public URL for asset upload."},
            "asset_type": {"type": "string", "enum": ["Image", "Video", "Audio"]},
            "group_type": {"type": "string"},
            "page_number": {"type": "integer"},
            "page_size": {"type": "integer"},
            "sort_by": {"type": "string"},
            "sort_order": {"type": "string", "enum": ["Asc", "Desc"]},
            "callback_url": {"type": "string"},
            "byted_token": {"type": "string"},
            "image_url": {"type": "string"},
            "images": {"type": "array", "items": {"type": "object", "properties": {"url": {"type": "string"}, "role": {"type": "string"}}}},
            "videos": {"type": "array", "items": {"type": "object", "properties": {"url": {"type": "string"}, "role": {"type": "string"}}}},
            "audios": {"type": "array", "items": {"type": "object", "properties": {"url": {"type": "string"}, "role": {"type": "string"}}}},
        },
        "required": ["action"],
    }
    is_concurrency_safe = False
    should_defer = False

    def __init__(self, client_provider: Callable[[], VideoGenerationClient | None]) -> None:
        self._client_provider = client_provider

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        client = self._client_provider()
        if client is None:
            return ToolResult(content="VideoApi failed: video API is not configured. Configure AGENT_VIDEO_BASE_URL and AGENT_VIDEO_TOKEN first.", is_error=True)
        action = str(tool_input.get("action") or "").strip()
        try:
            data = await self._dispatch(client, action, tool_input)
        except Exception as exc:
            return ToolResult(
                content=f"VideoApi {action or '<missing>'} failed:\n{format_exception_detail(exc)}",
                is_error=True,
                metadata={"action": action, "error_type": type(exc).__name__},
            )
        safe_data = _json_safe(data)
        return ToolResult(content=f"VideoApi {action} completed:\n{json.dumps(safe_data, ensure_ascii=False, indent=2)[:4000]}", metadata={"action": action, "response": safe_data})

    async def _dispatch(self, client: Any, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if not action:
            raise ValueError("action is required")
        if action == "rewrite_prompt":
            return await client.rewrite_prompt(
                prompt=str(data.get("prompt") or ""),
                pe_mode=str(data.get("pe_mode") or "creative"),
                rewrite_thinking_level=str(data.get("rewrite_thinking_level") or "standard"),
                duration=int(data.get("duration") or 8),
                image_url=str(data.get("image_url") or ""),
                images=_video_inputs(data.get("images"), "reference_image"),
                videos=_video_inputs(data.get("videos"), "reference_video"),
                audios=_video_inputs(data.get("audios"), "reference_audio"),
                wait=bool(data.get("wait", True)),
            )
        if action == "cancel_rewrite":
            return await client.cancel_rewrite(str(data.get("task_id") or ""))
        if action == "cancel_task":
            return await client.cancel_task(str(data.get("task_id") or ""))
        if action == "list_tasks":
            return await client.list_tasks(status=str(data.get("status") or ""), limit=int(data.get("limit") or 20), offset=int(data.get("offset") or 0))
        if action == "usage":
            return await client.usage(start_date=str(data.get("start_date") or ""), end_date=str(data.get("end_date") or ""), mode=str(data.get("mode") or ""))
        if action == "create_asset_group":
            return await client.create_asset_group(name=str(data.get("name") or ""), description=str(data.get("description") or ""))
        if action == "list_asset_groups":
            return await client.list_asset_groups(**_list_params(data))
        if action == "get_asset_group":
            return await client.get_asset_group(str(data.get("group_id") or ""))
        if action == "update_asset_group":
            return await client.update_asset_group(str(data.get("group_id") or ""), name=str(data.get("name") or ""), description=str(data.get("description") or ""))
        if action == "upload_asset":
            return await client.upload_asset(group_id=str(data.get("group_id") or ""), url=str(data.get("url") or ""), name=str(data.get("name") or ""), asset_type=str(data.get("asset_type") or "Image"))
        if action == "list_assets":
            return await client.list_assets(**_list_params(data))
        if action == "get_asset":
            return await client.get_asset(str(data.get("asset_id") or ""))
        if action == "update_asset":
            return await client.update_asset(str(data.get("asset_id") or ""), name=str(data.get("name") or ""))
        if action == "delete_asset":
            return await client.delete_asset(str(data.get("asset_id") or ""))
        if action == "create_validate_session":
            return await client.create_validate_session(callback_url=str(data.get("callback_url") or ""))
        if action == "get_validate_result":
            return await client.get_validate_result(byted_token=str(data.get("byted_token") or ""))
        raise ValueError(f"unsupported action: {action}")


def _video_inputs(value: Any, default_role: str) -> list[VideoInput]:
    if not isinstance(value, list):
        return []
    items: list[VideoInput] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        items.append(VideoInput(url=url, role=str(item.get("role") or default_role)))
    return items


def _list_params(data: dict[str, Any]) -> dict[str, Any]:
    keys = ("group_id", "status", "group_type", "name", "page_number", "page_size", "sort_by", "sort_order")
    return {key: data[key] for key in keys if data.get(key) not in (None, "", [])}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value
