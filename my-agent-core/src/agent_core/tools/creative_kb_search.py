from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_core.recovery.tool_errors import format_exception_detail
from agent_core.tools.base import ToolContext, ToolResult


DEFAULT_SEEDANCE_VEFAAS_PATH = "/Users/bytedance/PycharmProjects/empty/按照现在的故事板模式优化/seedance_vefaas"
DEFAULT_CREATIVE_KB_ENV_FILE = "/Users/bytedance/Desktop/.env"


class CreativeKBSearchV3Tool:
    name = "CreativeKBSearchV3"
    description = (
        "Search the Seedance ad creative knowledge-base v3 (ad_creative_kb_v3) and return reusable creative "
        "mechanisms, storyboard references, route diagnostics, and a rendered reference_context_text. Use it when "
        "planning ads, hooks, storyboards, or creative rewrites that need KB-backed inspiration."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Creative request or search prompt. Used to build body.prompt when body is omitted."},
            "body": {"type": "object", "description": "Raw CreativeKBSearchV3 body. Overrides convenience body fields."},
            "media_hints": {"type": "object", "description": "Optional product_identity/product_category hints from media understanding."},
            "intent_spec": {"type": "object", "description": "Structured intent: product_name, product_category, core_selling_points, style_requirements, etc."},
            "retrieval_filters": {"type": "object", "description": "Raw v3 retrieval filters, including creative_query_plan and creative_relevance_profile."},
            "product_name": {"type": "string"},
            "product_category": {"type": "string"},
            "industry": {"type": "string"},
            "core_selling_points": {"type": "array", "items": {"type": "string"}},
            "target_audience": {"type": "string"},
            "opening_requirements": {"type": "array", "items": {"type": "string"}},
            "style_requirements": {"type": "array", "items": {"type": "string"}},
            "creative_query_plan": {"type": "object", "description": "Fielded route queries: exact_queries, hook_queries, value_trust_queries, conversion_queries, structure_queries, style_queries."},
            "creative_relevance_profile": {"type": "object", "description": "Subject anchors and negative terms used by v3 relevance filtering."},
            "task_need": {"type": "string", "description": "storyboard, full_script, hook, product_demo, prompt_rewrite, or creative_idea."},
            "traffic_goal": {"type": "string", "description": "ad, natural, mixed, or free-form traffic hint."},
            "duration": {"type": "integer", "description": "Optional target duration in seconds."},
            "per_route_limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Raw recall limit per v3 route. Default from env or 8."},
            "selected_limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Final selected hit limit. Default from env or 5."},
            "max_context_chars": {"type": "integer", "minimum": 500, "maximum": 30000, "description": "Maximum reference_context_text chars included in tool text output."},
            "seedance_vefaas_path": {"type": "string", "description": "Path to the seedance_vefaas project. Defaults to AGENT_CREATIVE_KB_V3_SOURCE_DIR or known local path."},
            "env_file": {"type": "string", "description": "Dotenv file for VikingKB credentials. Defaults to AGENT_CREATIVE_KB_ENV_FILE or /Users/bytedance/Desktop/.env if present."},
        },
        "anyOf": [{"required": ["prompt"]}, {"required": ["body"]}],
    }
    is_concurrency_safe = True
    concurrency_group = "creative_kb_search_v3"
    max_concurrency = 3
    should_defer = False

    async def call(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        try:
            return await asyncio.to_thread(self._call_sync, dict(tool_input or {}))
        except Exception as exc:
            return ToolResult(
                content=f"CreativeKBSearchV3 failed:\n{format_exception_detail(exc)}",
                is_error=True,
                metadata={"error_type": type(exc).__name__},
            )

    def _call_sync(self, tool_input: dict[str, Any]) -> ToolResult:
        self._load_env(tool_input)
        service_cls = self._load_service_class(tool_input)
        per_route_limit = _bounded_int(
            tool_input.get("per_route_limit") or os.getenv("AGENT_CREATIVE_KB_V3_PER_ROUTE_LIMIT") or os.getenv("KB_STORYBOARD_V3_PER_ROUTE_LIMIT"),
            default=8,
            minimum=1,
            maximum=50,
        )
        selected_limit = _bounded_int(
            tool_input.get("selected_limit") or os.getenv("AGENT_CREATIVE_KB_V3_SELECTED_LIMIT") or os.getenv("KB_STORYBOARD_V3_SELECTED_LIMIT"),
            default=5,
            minimum=1,
            maximum=20,
        )
        service = service_cls(per_route_limit=per_route_limit, selected_limit=selected_limit)
        body, media_hints, intent_spec, retrieval_filters = _build_search_inputs(tool_input)
        result = service.search(
            body=body,
            media_hints=media_hints,
            intent_spec=intent_spec,
            retrieval_filters=retrieval_filters,
        )
        payload = _result_to_payload(result)
        return ToolResult(
            content=_render_tool_output(payload, max_context_chars=_bounded_int(tool_input.get("max_context_chars"), default=8000, minimum=500, maximum=30000)),
            metadata=payload,
        )

    @staticmethod
    def _load_env(tool_input: dict[str, Any]) -> None:
        raw = (
            tool_input.get("env_file")
            or os.getenv("AGENT_CREATIVE_KB_ENV_FILE")
            or os.getenv("CREATIVE_KB_ENV_FILE")
            or (DEFAULT_CREATIVE_KB_ENV_FILE if Path(DEFAULT_CREATIVE_KB_ENV_FILE).exists() else "")
        )
        if raw:
            load_dotenv(Path(str(raw)).expanduser(), override=True)

    @staticmethod
    def _load_service_class(tool_input: dict[str, Any]) -> Any:
        source_path = Path(str(
            tool_input.get("seedance_vefaas_path")
            or os.getenv("AGENT_CREATIVE_KB_V3_SOURCE_DIR")
            or os.getenv("SEEDANCE_VEFAAS_DIR")
            or DEFAULT_SEEDANCE_VEFAAS_PATH
        )).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(
                "seedance_vefaas source path not found; set AGENT_CREATIVE_KB_V3_SOURCE_DIR or pass seedance_vefaas_path"
            )
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            module = importlib.import_module("services.hot_v_creative_kb_service")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"failed to import seedance v3 search from {source_path}; ensure dependencies are installed: {exc}"
            ) from exc
        return getattr(module, "CreativeKBSearchV3")


def _build_search_inputs(tool_input: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = str(tool_input.get("prompt") or "").strip()
    body = _dict_value(tool_input.get("body"))
    if not body:
        body = {"prompt": prompt}
    elif prompt and not body.get("prompt"):
        body["prompt"] = prompt
    task_need = str(tool_input.get("task_need") or body.get("task_need") or "storyboard").strip()
    if task_need:
        body.setdefault("task_need", task_need)
        body.setdefault("pe_mode", "storyboard" if task_need == "storyboard" else task_need)
    if tool_input.get("traffic_goal") is not None:
        body.setdefault("traffic_goal", tool_input.get("traffic_goal"))
    if tool_input.get("duration") is not None:
        body.setdefault("duration", tool_input.get("duration"))

    media_hints = _dict_value(tool_input.get("media_hints"))
    intent_spec = _dict_value(tool_input.get("intent_spec"))
    _merge_if_present(intent_spec, tool_input, "product_name")
    _merge_if_present(intent_spec, tool_input, "product_category")
    _merge_if_present(intent_spec, tool_input, "industry")
    _merge_if_present(intent_spec, tool_input, "core_selling_points")
    _merge_if_present(intent_spec, tool_input, "target_audience")
    _merge_if_present(intent_spec, tool_input, "opening_requirements")
    _merge_if_present(intent_spec, tool_input, "style_requirements")
    if tool_input.get("product_name") and not media_hints.get("product_identity"):
        media_hints["product_identity"] = tool_input.get("product_name")
    if tool_input.get("product_category") and not media_hints.get("product_category"):
        media_hints["product_category"] = tool_input.get("product_category")

    retrieval_filters = _dict_value(tool_input.get("retrieval_filters"))
    if tool_input.get("traffic_goal") is not None:
        retrieval_filters.setdefault("traffic_goal", tool_input.get("traffic_goal"))
    creative_query_plan = _dict_value(tool_input.get("creative_query_plan")) or _dict_value(retrieval_filters.get("creative_query_plan"))
    if not creative_query_plan:
        creative_query_plan = _default_creative_query_plan(body, intent_spec)
    if creative_query_plan:
        retrieval_filters["creative_query_plan"] = creative_query_plan
    creative_relevance_profile = _dict_value(tool_input.get("creative_relevance_profile"))
    if creative_relevance_profile:
        retrieval_filters["creative_relevance_profile"] = creative_relevance_profile
    return body, media_hints, intent_spec, retrieval_filters


def _default_creative_query_plan(body: dict[str, Any], intent_spec: dict[str, Any]) -> dict[str, list[str]]:
    prompt = _clean_join([body.get("prompt")], limit=180)
    product = _clean_join([intent_spec.get("product_category"), intent_spec.get("product_name"), intent_spec.get("industry")], limit=120)
    selling = _clean_join([intent_spec.get("core_selling_points")], limit=120)
    style = _clean_join([intent_spec.get("style_requirements")], limit=120)
    anchor = _clean_join([product, selling, prompt], limit=180)
    return {
        "exact_queries": _non_empty([anchor or prompt]),
        "hook_queries": _non_empty([_clean_join([product, selling, "前3秒 抓停 可见画面转译", prompt], limit=180)]),
        "value_trust_queries": _non_empty([_clean_join([product, selling, "可见证明动作 使用场景 结果状态"], limit=160)]),
        "structure_queries": _non_empty([_clean_join([anchor, "15秒完整链路 故事板 首格 中段 结尾", style], limit=180)]),
        "style_queries": _non_empty([style]),
    }


def _result_to_payload(result: Any) -> dict[str, Any]:
    hits = list(getattr(result, "hits", None) or [])
    hit_frames = list(getattr(result, "hit_frames", None) or [])
    reference_context_text = str(getattr(result, "reference_context_text", "") or "")
    search_meta = dict(getattr(result, "search_meta", None) or {})
    return {
        "hits": hits,
        "hit_frames": hit_frames,
        "reference_context_text": reference_context_text,
        "search_meta": search_meta,
    }


def _render_tool_output(payload: dict[str, Any], *, max_context_chars: int) -> str:
    meta = dict(payload.get("search_meta") or {})
    routes = list(meta.get("routes") or [])
    lines = [
        "CreativeKBSearchV3 completed.",
        f"- provider: {meta.get('provider') or meta.get('strategy') or 'ad_creative_kb_v3'}",
        f"- raw/merged/filtered/selected: {meta.get('raw_hit_count', 0)}/{meta.get('merged_hit_count', 0)}/{meta.get('filtered_hit_count', 0)}/{meta.get('selected_hit_count', len(payload.get('hits') or []))}",
    ]
    route_errors = list(meta.get("route_errors") or [])
    if route_errors:
        lines.append(f"- route_errors: {json.dumps(route_errors, ensure_ascii=False)}")
    if routes:
        lines.append("- routes:")
        for route in routes:
            lines.append(
                f"  - {route.get('name')}[{route.get('doc_type')}]: hits={route.get('hit_count', 0)}, fallback={route.get('fallback_used') or '-'}, error={route.get('error') or '-'}"
            )
    hits = list(payload.get("hits") or [])
    if hits:
        lines.append("- selected_hits:")
        for idx, hit in enumerate(hits[:8], start=1):
            title = hit.get("title") or hit.get("item_title") or hit.get("item_id") or hit.get("doc_id") or "untitled"
            level = hit.get("creative_relevance_level") or ""
            score = hit.get("creative_relevance_score") or hit.get("score") or ""
            lines.append(f"  {idx}. {title} | relevance={level}/{score} | route={hit.get('route_name') or hit.get('matched_routes') or '-'}")
    context_text = str(payload.get("reference_context_text") or "")
    if context_text:
        clipped = context_text[:max_context_chars]
        suffix = "\n...[truncated]" if len(context_text) > max_context_chars else ""
        lines.append("\nreference_context_text:\n" + clipped + suffix)
    return "\n".join(lines)


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _merge_if_present(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    if source.get(key) not in (None, "", []):
        target.setdefault(key, source[key])


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    with contextlib.suppress(TypeError, ValueError):
        parsed = int(value)
        return max(minimum, min(maximum, parsed))
    return default


def _clean_join(values: list[Any], *, limit: int) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            iterable = [value]
        for item in iterable:
            text = str(item or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)[:limit].strip()


def _non_empty(values: list[str]) -> list[str]:
    return [value for value in values if str(value or "").strip()]
