from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from agent_core.tools.base import ToolContext
from agent_core.tools.creative_kb_search import CreativeKBSearchV3Tool


@dataclass
class _FakeResult:
    hits: list[dict]
    hit_frames: list[dict]
    reference_context_text: str
    search_meta: dict


@pytest.mark.asyncio
async def test_creative_kb_search_v3_tool_builds_inputs_and_returns_metadata(tmp_path, monkeypatch):
    captured: dict = {}

    class FakeCreativeKBSearchV3:
        def __init__(self, *, per_route_limit: int, selected_limit: int) -> None:
            captured["limits"] = (per_route_limit, selected_limit)

        def search(self, *, body, media_hints, intent_spec, retrieval_filters):
            captured["body"] = body
            captured["media_hints"] = media_hints
            captured["intent_spec"] = intent_spec
            captured["retrieval_filters"] = retrieval_filters
            return _FakeResult(
                hits=[
                    {
                        "title": "护肤水润质感首帧",
                        "route_name": "asset_hook",
                        "creative_relevance_level": "strong",
                        "creative_relevance_score": 0.82,
                    }
                ],
                hit_frames=[{"url": "https://example.com/frame.jpg", "label": "首帧"}],
                reference_context_text="## V3创意简报\n可借水润质感和结果前置。",
                search_meta={
                    "provider": "ad_creative_kb_v3",
                    "raw_hit_count": 3,
                    "merged_hit_count": 2,
                    "filtered_hit_count": 1,
                    "selected_hit_count": 1,
                    "route_errors": [],
                    "routes": [
                        {"name": "asset_hook", "doc_type": "creative_asset", "hit_count": 1, "error": "", "fallback_used": ""}
                    ],
                },
            )

    fake_module = types.ModuleType("services.hot_v_creative_kb_service")
    fake_module.CreativeKBSearchV3 = FakeCreativeKBSearchV3
    monkeypatch.setitem(sys.modules, "services.hot_v_creative_kb_service", fake_module)

    tool = CreativeKBSearchV3Tool()
    result = await tool.call(
        {
            "prompt": "给百雀羚做15秒护肤故事板",
            "product_name": "百雀羚",
            "product_category": "护肤品",
            "core_selling_points": ["保湿", "修护"],
            "creative_query_plan": {"hook_queries": ["护肤 前3秒 水润抓停"]},
            "per_route_limit": 3,
            "selected_limit": 2,
            "seedance_vefaas_path": str(tmp_path),
            "env_file": str(tmp_path / "missing.env"),
        },
        ToolContext(session_id="s1", messages=[]),
    )

    assert result.is_error is False
    assert "CreativeKBSearchV3 completed" in result.content
    assert "raw/merged/filtered/selected: 3/2/1/1" in result.content
    assert result.metadata["search_meta"]["provider"] == "ad_creative_kb_v3"
    assert captured["limits"] == (3, 2)
    assert captured["body"]["prompt"] == "给百雀羚做15秒护肤故事板"
    assert captured["body"]["task_need"] == "storyboard"
    assert captured["media_hints"]["product_identity"] == "百雀羚"
    assert captured["intent_spec"]["product_category"] == "护肤品"
    assert captured["retrieval_filters"]["creative_query_plan"]["hook_queries"] == ["护肤 前3秒 水润抓停"]


@pytest.mark.asyncio
async def test_creative_kb_search_v3_tool_reports_missing_source_path(tmp_path):
    tool = CreativeKBSearchV3Tool()
    result = await tool.call(
        {
            "prompt": "检索护肤广告创意",
            "seedance_vefaas_path": str(tmp_path / "does-not-exist"),
            "env_file": str(tmp_path / "missing.env"),
        },
        ToolContext(session_id="s1", messages=[]),
    )

    assert result.is_error is True
    assert "seedance_vefaas source path not found" in result.content
