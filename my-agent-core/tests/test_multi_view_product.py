"""
端到端测试：图片理解 → 多视角商品设定稿生成

流程：
1. 用 GenerateImage 生成一张测试商品图（洗发水瓶）
2. 用 UnderstandImage（豆包 Vision）观察商品细节
3. 用 GenerateImage 画出多视角设定稿（正面/45度/侧面/背面）
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

# 确保项目在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from agent_core.attachments import ImageAttachmentStore
from agent_core.images.top_aidp import TopAidpImageGenerationClient
from agent_core.vision.openai_compatible import OpenAICompatibleVisionClient
import os


OUTPUT_DIR = Path(__file__).resolve().parent / "_multi_view_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def build_image_client() -> TopAidpImageGenerationClient:
    return TopAidpImageGenerationClient(
        app_key=_env("TOP_AIDP_APP_KEY"),
        app_secret=_env("TOP_AIDP_APP_SECRET"),
        base_url=_env("TOP_AIDP_BASE_URL"),
        model=_env("AGENT_IMAGE_MODEL") or _env("TOP_AIDP_MODEL"),
        api_version=_env("TOP_AIDP_VERSION") or "2022-08-01",
        create_method=_env("TOP_AIDP_CREATE_METHOD") or "CreateImageTask",
        status_method=_env("TOP_AIDP_STATUS_METHOD") or "GetImageTaskStatus",
        timeout=float(_env("TOP_AIDP_TIMEOUT") or "120"),
        poll_interval=float(_env("TOP_AIDP_POLL_INTERVAL") or "2"),
        max_polls=int(_env("TOP_AIDP_MAX_POLLS") or "90"),
        output_url=(_env("TOP_AIDP_RETURN_URL") or "true").strip().lower() not in {"0", "false", "no", "off"},
        signature_protocol=_env("TOP_AIDP_SIGNATURE_PROTOCOL") or "volsign",
        volc_service=_env("TOP_AIDP_VOLC_SERVICE") or "cdp_saas",
        volc_region=_env("TOP_AIDP_VOLC_REGION") or "cn-beijing",
        aidp_token=_env("TOP_AIDP_TOKEN"),
    )


def build_vision_client() -> OpenAICompatibleVisionClient:
    model = _env("AGENT_VISION_MODEL")
    api_key = _env("AGENT_VISION_API_KEY") or _env("ARK_API_KEY")
    base_url = _env("AGENT_VISION_BASE_URL") or _env("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    return OpenAICompatibleVisionClient(model=model, api_key=api_key, base_url=base_url)


def save_image(data: bytes, name: str) -> str:
    path = OUTPUT_DIR / name
    path.write_bytes(data)
    return str(path)


MULTI_VIEW_PROMPT = """这是一张商品结构复原参考图。请把输入图中的同一商品画成多视角设定稿，用于让后续视频模型稳定保持商品身份。

先仔细观察输入商品的外轮廓、长宽比例、瓶盖/泵头/封口形状、瓶肩和底部结构、标签区域位置、主色、材质分区、清晰可读的品牌或产品文字；再结合该品类常识，合理推断侧面厚度、45度视角和背面的大致包装结构。

画面在一张图内并排展示正面、45度侧前、侧面、背面/顶部关键结构，所有视角必须是同一商品，轮廓比例、瓶盖结构、标签位置、主色和材质连续一致。

输入图中清晰可见的 logo/主标题/大字按原位置和拼写保留；不可读小字不要猜，画成不可读的短线或色块。

中性背景、稳定棚拍光、无人物、无道具、无场景、无广告氛围。不要添加箭头、标注线、视角名称、说明文字、水印、假品牌或新文案。不要重新设计商品，不要换包装，不要改变品类。"""


async def main() -> None:
    print("=" * 70)
    print("Step 1: 生成测试商品图（洗发水瓶）")
    print("=" * 70)

    image_client = build_image_client()
    vision_client = build_vision_client()

    print(f"  Image provider: top_aidp (token={_env('TOP_AIDP_TOKEN')[:20]}...)")
    print(f"  Vision model: {_env('AGENT_VISION_MODEL')}")

    # Step 1: 生成一张测试商品图
    product_prompt = (
        "A single shampoo bottle product photo, front view, studio lighting, neutral white background. "
        "The bottle is tall cylindrical shape with a white pump dispenser on top, gradient blue-to-cyan body, "
        "silver metallic cap ring, rounded shoulders. A white label wraps around the middle with the brand name "
        "'AQUA PURE' in bold navy blue letters, subtitle 'Hydrating Shampoo' in smaller text below. "
        "The bottle has a slight curve at the bottom. No hands, no props, no scene, no reflections on floor. "
        "Clean product photography style."
    )

    print("  Generating product image...")
    result = await image_client.generate_image(
        prompt=product_prompt,
        size="1024x1024",
        n=1,
    )

    product_image = result.images[0]
    product_path = save_image(product_image.data, "01_product_input.png")
    print(f"  Saved: {product_path}")
    print(f"  Revised prompt: {product_image.revised_prompt[:200]}...")
    print()

    # Step 2: 用 Vision 模型理解商品
    print("=" * 70)
    print("Step 2: UnderstandImage — 分析商品细节")
    print("=" * 70)

    vision_prompt = (
        "Analyze this product image in extreme detail for multi-view reference sheet creation. "
        "Describe:\n"
        "1. Outer silhouette and aspect ratio (height:width)\n"
        "2. Cap/pump/dispenser shape, color, material\n"
        "3. Shoulder and bottom structure\n"
        "4. Label area position, shape, colors\n"
        "5. All readable brand/product text (exact spelling, position, font color, size)\n"
        "6. Material zones (glossy/matte/transparent/metallic)\n"
        "7. Dominant colors with hex approximations\n"
        "8. Any decorative elements or patterns\n\n"
        "Be specific and quantitative. This description will be used as input for generating "
        "a multi-view orthographic reference sheet."
    )

    print("  Sending to vision model...")
    vision_result = await vision_client.understand_image(
        image_bytes=product_image.data,
        content_type=product_image.content_type,
        prompt=vision_prompt,
    )
    print(f"  Vision analysis:\n{vision_result.text}")
    print()

    # Step 3: 生成多视角设定稿
    print("=" * 70)
    print("Step 3: GenerateImage — 多视角设定稿")
    print("=" * 70)

    # 把 vision 分析结果和原图信息合并到 prompt 中
    multi_view_full_prompt = f"""{MULTI_VIEW_PROMPT}

Product analysis from vision model:
{vision_result.text}

Based on the above analysis and the original product image, create the multi-view reference sheet."""

    print("  Generating multi-view reference sheet...")
    multi_result = await image_client.generate_image(
        prompt=multi_view_full_prompt,
        size="1536x1024",  # landscape for side-by-side views
        n=1,
    )

    multi_path = save_image(multi_result.images[0].data, "02_multi_view_output.png")
    print(f"  Saved: {multi_path}")
    print(f"  Revised prompt: {multi_result.revised_prompt[:300]}...")
    print()

    # Step 4: 对比分析
    print("=" * 70)
    print("Step 4: 对比分析 — 检查多视角一致性")
    print("=" * 70)

    compare_prompt = (
        "Compare the multi-view reference sheet (which should show front, 45-degree, side, and back views "
        "of the same product) against the original single product image. Evaluate:\n"
        "1. Are all views clearly the same product? (silhouette, proportions, cap structure)\n"
        "2. Is the brand text 'AQUA PURE' preserved correctly across views?\n"
        "3. Are colors and materials consistent?\n"
        "4. Are label positions consistent?\n"
        "5. Is the background neutral with no arrows/labels/watermarks?\n"
        "6. Overall quality assessment for video model reference use.\n\n"
        "Be honest and critical. Point out any inconsistencies or issues."
    )

    print("  Sending comparison to vision model...")
    compare_result = await vision_client.understand_image(
        image_bytes=multi_result.images[0].data,
        content_type=multi_result.images[0].content_type,
        prompt=compare_prompt,
    )
    print(f"  Comparison analysis:\n{compare_result.text}")
    print()

    # 保存完整报告
    report = {
        "product_prompt": product_prompt,
        "product_revised_prompt": product_image.revised_prompt,
        "vision_analysis": vision_result.text,
        "multi_view_prompt": multi_view_full_prompt,
        "multi_view_revised_prompt": multi_result.revised_prompt,
        "comparison_analysis": compare_result.text,
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Full report saved: {report_path}")
    print()
    print("=" * 70)
    print("Done! Check _multi_view_output/ for:")
    print("  01_product_input.png   — 原始测试商品图")
    print("  02_multi_view_output.png — 多视角设定稿")
    print("  report.json            — 完整分析报告")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
