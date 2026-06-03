from __future__ import annotations

import os
import uuid
import contextlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_core.attachments import ImageAttachmentStore
from agent_core.billing.store import BillingStore
from agent_core.buddy import build_companion_prompt
from agent_core.context.builder import ContextBuilder, system_prompt_section, uncached_system_prompt_section
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.hooks.store import HookStore
from agent_core.images import ImageGenerationClient
from agent_core.integrations.feishu import FeishuApiTool, FeishuTokenStore
from agent_core.memory.session_memory import SessionMemoryConfig, SessionMemoryManager
from agent_core.model.base import ModelClient, ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.permissions.policy import StaticPermissionPolicy
from agent_core.quota.store import QuotaStore
from agent_core.skills.store import SkillStore, build_skill_prompt_section
from agent_core.session.store import create_session_store, SessionStore
from agent_core.tasks import FileTaskStore, TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool
from agent_core.teams import AgentTool, ReadInboxTool, SendMessageTool, TeamCreateTool, TeamDeleteTool, TeamListTool, TeamStore
from agent_core.tools.base import ToolRegistry
from agent_core.tools.builtin import (
    EchoTool,
    ToolSearchTool,
    TodoWriteTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    ReadTextFileTool,
    WriteTextFileTool,
    EditFileTool,
    ListDirectoryTool,
    GrepTool,
    GlobTool,
    BashTool,
    BashOutputTool,
    KillShellTool,
)
from agent_core.tools.creative_kb_search import CreativeKBSearchV3Tool
from agent_core.tools.image_generation import GenerateImageTool
from agent_core.tools.image_understanding import UnderstandImageTool
from agent_core.tools.video_api import VideoApiTool
from agent_core.tools.video_generation import GenerateVideoTool
from agent_core.tools.skill import SkillTool
from agent_core.tools.subagent import TaskTool
from agent_core.tools.web import WebFetchTool, WebSearchTool
from agent_core.types import TextBlock, ThinkingBlock, ToolUseBlock
from agent_core.users.store import UserStore
from agent_core.videos import VideoGenerationClient
from agent_core.vision.base import VisionClient
from agent_core.worktree import WorktreeManager
from agent_core.plan_mode import plan_mode_system_prompt
from agent_server.codex_credentials import codex_base_url, codex_image_model, resolve_codex_provider_credentials
from agent_server.model_providers import create_model_from_env, default_model_for_provider, detect_provider_from_env
from agent_server.prompt_store import PromptStore

# 自动加载 .env — 从项目根目录（src 的父目录）查找
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()
_env_path = _PROJECT_ROOT / ".env"
load_dotenv(_env_path, override=True)
_task_store = FileTaskStore(task_list_id="tasklist")
_team_store = TeamStore()


_attachment_store = ImageAttachmentStore(
    Path(os.getenv("AGENT_ATTACHMENTS_DIR") or Path.home() / ".my-agent-core" / "attachments"),
    max_size_bytes=int(
        os.getenv("AGENT_MAX_ATTACHMENT_BYTES")
        or os.getenv("AGENT_MAX_IMAGE_ATTACHMENT_BYTES")
        or str(50 * 1024 * 1024)
    ),
)


def get_task_store() -> FileTaskStore:
    return _task_store


def get_team_store() -> TeamStore:
    return _team_store


def get_attachment_store() -> ImageAttachmentStore:
    return _attachment_store


_image_generation_client: ImageGenerationClient | None | object = None
_video_generation_client: VideoGenerationClient | None | object = None


def set_image_generation_client_for_tests(client: ImageGenerationClient | None) -> None:
    global _image_generation_client
    _image_generation_client = client


def clear_image_generation_client_cache() -> None:
    global _image_generation_client
    _image_generation_client = None


def clear_video_generation_client_cache() -> None:
    global _video_generation_client
    _video_generation_client = None


def get_image_generation_client(provider: str | None = None) -> ImageGenerationClient | None:
    global _image_generation_client
    if _image_generation_client is not None:
        return _image_generation_client if _image_generation_client else None  # type: ignore[return-value]
    selected_provider = provider or _detect_image_provider_from_env()
    if not selected_provider:
        _image_generation_client = None
        return None
    selected_provider = selected_provider.lower()
    _image_generation_client = _create_image_generation_client_from_env(selected_provider)
    return _image_generation_client  # type: ignore[return-value]


def get_video_generation_client() -> VideoGenerationClient | None:
    global _video_generation_client
    if _video_generation_client is not None:
        return _video_generation_client if _video_generation_client else None  # type: ignore[return-value]
    base_url = os.getenv("AGENT_VIDEO_BASE_URL") or os.getenv("VIDEO_BASE_URL")
    token = os.getenv("AGENT_VIDEO_TOKEN") or os.getenv("VIDEO_BEARER_TOKEN")
    if not base_url or not token:
        _video_generation_client = None
        return None
    try:
        from agent_core.videos.rest_bearer import BearerVideoGenerationClient
        _video_generation_client = BearerVideoGenerationClient(
            base_url=base_url,
            token=token,
            timeout=float(os.getenv("AGENT_VIDEO_TIMEOUT") or "30"),
            poll_interval=float(os.getenv("AGENT_VIDEO_POLL_INTERVAL") or "10"),
            max_polls=int(os.getenv("AGENT_VIDEO_MAX_POLLS") or "120"),
        )
    except (ImportError, ValueError):
        _video_generation_client = None
    return _video_generation_client  # type: ignore[return-value]


def _detect_image_provider_from_env() -> str | None:
    """Resolve image generation provider independently from the chat model.

    Image generation is a tool/playground capability.  It must never change or
    inherit the main chat provider implicitly, otherwise configuring Codex image
    credentials can break ordinary conversation.
    """
    explicit = (os.getenv("AGENT_IMAGE_PROVIDER") or os.getenv("IMAGE_PROVIDER") or "").strip().lower()
    if explicit and explicit != "auto":
        return explicit
    if os.getenv("TOP_AIDP_BASE_URL") and (os.getenv("TOP_AIDP_APP_KEY") or os.getenv("TOP_APP_KEY")) and (os.getenv("TOP_AIDP_APP_SECRET") or os.getenv("TOP_APP_SECRET")):
        return "top_aidp"
    if os.getenv("AGENT_IMAGE_MODEL") or os.getenv("AGENT_IMAGE_BASE_URL") or os.getenv("AGENT_IMAGE_API_KEY"):
        return "image"
    if resolve_codex_provider_credentials():
        return "codex"
    if os.getenv("OPENAI_IMAGE_MODEL") and os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def create_worktree_manager(*, cwd: str | Path | None = None) -> WorktreeManager:
    try:
        hook_engine = _hook_store.create_engine("default")
    except FileNotFoundError:
        from agent_core.hooks.engine import HookEngine
        hook_engine = HookEngine()
    return WorktreeManager(cwd=cwd or _WORKSPACE_ROOT, hook_engine=hook_engine)


# ── 从环境变量创建 ModelClient ────────────────────────────────


def _create_model_from_env(provider: str, *, model_id: str | None = None) -> ModelClient:
    """根据 provider 名称从环境变量创建 ModelClient。"""
    return create_model_from_env(provider, model_id=model_id)


def _model_id_for_provider(provider: str, *, model_id: str | None = None) -> str:
    """Resolve the model id with the same precedence used to create clients."""
    return default_model_for_provider(provider, model_id=model_id)


def _detect_provider_from_env() -> str:
    return detect_provider_from_env()


def _create_vision_client_from_env(provider: str) -> VisionClient | None:
    """Create a vision client when a vision-capable model is configured."""
    if provider.startswith("fake"):
        return None
    explicit_model = os.getenv("AGENT_VISION_MODEL") or os.getenv("VISION_MODEL")
    api_key = os.getenv("AGENT_VISION_API_KEY")
    base_url = os.getenv("AGENT_VISION_BASE_URL")
    model = explicit_model
    if provider == "openai":
        model = model or os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
    elif provider in ("ark", "volcengine", "doubao"):
        model = model or os.getenv("ARK_VISION_MODEL") or os.getenv("ARK_MODEL") or os.getenv("MODEL_ID")
        api_key = api_key or os.getenv("ARK_API_KEY")
        base_url = base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    elif provider == "gateway":
        model = model or os.getenv("GATEWAY_VISION_MODEL") or os.getenv("GATEWAY_MODEL") or os.getenv("MODEL_ID")
        api_key = api_key or os.getenv("GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("GATEWAY_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    elif provider == "codex":
        codex_credentials = resolve_codex_provider_credentials()
        model = model or os.getenv("CODEX_VISION_MODEL") or os.getenv("CODEX_MODEL") or os.getenv("MODEL_ID")
        api_key = api_key or (codex_credentials.api_key if codex_credentials else None) or os.getenv("GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or codex_base_url()
    if not model:
        return None
    try:
        from agent_core.vision.openai_compatible import OpenAICompatibleVisionClient
        return OpenAICompatibleVisionClient(model=model, api_key=api_key or "", base_url=base_url)
    except ImportError:
        return None


def _create_image_generation_client_from_env(provider: str) -> ImageGenerationClient | None:
    """Create an image generation/edit client from environment configuration."""
    if provider in ("top", "top_aidp", "aidp", "cdp", "volc_top"):
        try:
            from agent_core.images.top_aidp import TopAidpImageGenerationClient
            return TopAidpImageGenerationClient(
                app_key=os.getenv("TOP_AIDP_APP_KEY") or os.getenv("AIDP_APP_KEY") or os.getenv("TOP_APP_KEY") or "",
                app_secret=os.getenv("TOP_AIDP_APP_SECRET") or os.getenv("AIDP_APP_SECRET") or os.getenv("TOP_APP_SECRET") or "",
                base_url=os.getenv("TOP_AIDP_BASE_URL") or os.getenv("AIDP_BASE_URL") or os.getenv("TOP_BASE_URL") or "",
                model=os.getenv("AGENT_IMAGE_MODEL") or os.getenv("TOP_AIDP_MODEL") or os.getenv("AIDP_IMAGE_MODEL") or os.getenv("IMAGE_MODEL") or "",
                api_version=os.getenv("TOP_AIDP_VERSION") or os.getenv("TOP_API_VERSION") or "2.0",
                create_method=os.getenv("TOP_AIDP_CREATE_METHOD") or "CreateImageTask",
                status_method=os.getenv("TOP_AIDP_STATUS_METHOD") or "GetImageTaskStatus",
                timeout=float(os.getenv("TOP_AIDP_TIMEOUT") or os.getenv("AGENT_IMAGE_TIMEOUT") or "120"),
                poll_interval=float(os.getenv("TOP_AIDP_POLL_INTERVAL") or "2"),
                max_polls=int(os.getenv("TOP_AIDP_MAX_POLLS") or "90"),
                output_url=(os.getenv("TOP_AIDP_RETURN_URL") or "true").strip().lower() not in {"0", "false", "no", "off"},
                signature_protocol=os.getenv("TOP_AIDP_SIGNATURE_PROTOCOL") or os.getenv("TOP_AIDP_SIGNATURE") or "top",
                volc_service=os.getenv("TOP_AIDP_VOLC_SERVICE") or "cdp_saas",
                volc_region=os.getenv("TOP_AIDP_VOLC_REGION") or "cn-beijing",
                aidp_token=os.getenv("TOP_AIDP_TOKEN") or os.getenv("AIDP_IMAGE_TOKEN") or os.getenv("STORYBOARD_AIDP_IMAGE_TOKEN") or "",
                session_token=os.getenv("TOP_AIDP_SESSION_TOKEN") or os.getenv("VOLCENGINE_SESSION_TOKEN") or "",
            )
        except (ImportError, ValueError):
            return None
    model = os.getenv("AGENT_IMAGE_MODEL") or os.getenv("IMAGE_MODEL")
    api_key = os.getenv("AGENT_IMAGE_API_KEY")
    base_url = os.getenv("AGENT_IMAGE_BASE_URL")
    api_mode = (os.getenv("AGENT_IMAGE_API_MODE") or "images").strip().lower()
    codex_cli = (os.getenv("AGENT_IMAGE_CODEX_CLI") or os.getenv("IMAGE_CODEX_CLI") or "").strip().lower() in {"1", "true", "yes", "on"}
    if provider == "image":
        model = model or os.getenv("OPENAI_IMAGE_MODEL") or os.getenv("CODEX_IMAGE_MODEL") or "gpt-image-2"
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY") or os.getenv("GATEWAY_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("CODEX_BASE_URL") or os.getenv("GATEWAY_BASE_URL") or "https://api.openai.com/v1"
    elif provider == "openai":
        model = model or os.getenv("OPENAI_IMAGE_MODEL") or "gpt-image-2"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    elif provider in ("ark", "volcengine", "doubao"):
        model = model or os.getenv("ARK_IMAGE_MODEL") or os.getenv("ARK_MODEL") or os.getenv("MODEL_ID")
        api_key = api_key or os.getenv("ARK_API_KEY")
        base_url = base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    elif provider == "gateway":
        model = model or os.getenv("GATEWAY_IMAGE_MODEL") or os.getenv("GATEWAY_MODEL") or os.getenv("OPENAI_IMAGE_MODEL") or "gpt-image-2"
        api_key = api_key or os.getenv("GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("GATEWAY_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    elif provider == "codex":
        codex_credentials = resolve_codex_provider_credentials()
        model = model or codex_image_model()
        api_key = api_key or (codex_credentials.api_key if codex_credentials else None) or os.getenv("GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or codex_base_url()
    if not model:
        return None
    try:
        from agent_core.images.openai_compatible import OpenAICompatibleImageGenerationClient
        return OpenAICompatibleImageGenerationClient(
            model=model,
            api_key=api_key or "",
            base_url=base_url,
            api_mode=api_mode,
            codex_cli=codex_cli,
        )
    except ImportError:
        return None


def get_session_model_selection(session_id: str | None) -> dict[str, str]:
    """返回 session 级模型选择；没有选择时回退到环境变量/自动探测。"""
    metadata: dict[str, Any] = {}
    if session_id:
        try:
            info = _session_store.get_session(session_id)
            if info and isinstance(info.get("metadata"), dict):
                metadata = info["metadata"]
        except Exception:
            metadata = {}
    provider = str(metadata.get("model_provider") or "").strip().lower()
    model_id = str(metadata.get("model_id") or "").strip()
    if not provider or provider == "auto":
        provider = _detect_provider_from_env()
    elif provider == "codex" and resolve_codex_provider_credentials() is None:
        # Older sessions may have been switched to codex by image/Codex login
        # experiments.  Do not let an image-only or official auth.json login
        # break the main chat chain.
        provider = _detect_provider_from_env()
        model_id = ""
    return {
        "model_provider": provider,
        "model_id": _model_id_for_provider(provider, model_id=model_id or None),
    }


def default_prompt_template_name() -> str:
    return (os.getenv("AGENT_PROMPT_TEMPLATE") or os.getenv("PROMPT_TEMPLATE") or "default").strip() or "default"


def get_session_prompt_template(session_id: str | None, *, explicit_template_name: str | None = None) -> str:
    """Resolve the prompt template for a session.

    Precedence: explicit caller override → session metadata → environment default.
    """
    if explicit_template_name:
        return explicit_template_name.strip() or default_prompt_template_name()
    metadata: dict[str, Any] = {}
    if session_id:
        try:
            info = _session_store.get_session(session_id)
            if info and isinstance(info.get("metadata"), dict):
                metadata = info["metadata"]
        except Exception:
            metadata = {}
    selected = str(metadata.get("prompt_template") or metadata.get("template_name") or "").strip()
    return selected or default_prompt_template_name()


def list_prompt_templates() -> list[dict[str, Any]]:
    return _prompt_store.list_templates()


# ── PromptStore 单例 ──────────────────────────────────────────

_prompt_store = PromptStore()

# ── HookStore 单例 ────────────────────────────────────────────

_hook_store = HookStore(hooks_dir=Path(__file__).resolve().parent / "hooks")

# ── SessionStore 单例（工厂创建，配置驱动）──────────────────────

_session_store: SessionStore = create_session_store()

# ── BillingStore 单例（模型价格表 + session usage）───────────────

_billing_store = BillingStore()

# ── User/Quota 单例（Claude Code accountUuid/organizationUuid 对齐）────────

_user_store = UserStore()
_quota_store = QuotaStore()

# ── Feishu integration（按 user_id + org_id 存储 User Access Token）──────────

_feishu_token_store = FeishuTokenStore()

# ── SkillStore 单例 ───────────────────────────────────────────

_skill_store = SkillStore(cwd=_WORKSPACE_ROOT)


# ── 创建 Runtime ─────────────────────────────────────────────


def create_runtime(
    *,
    mode: str | None = None,
    session_id: str | None = None,
    template_name: str | None = None,
    ask_callback=None,
    user_id: str | None = None,
    org_id: str | None = None,
    cwd: str | Path | None = None,
) -> AgentRuntime:
    """创建 AgentRuntime。

    模型选择逻辑（优先级从高到低）：
    1. mode 参数（代码显式指定 provider）
    2. session metadata 中的 model_provider/model_id（由 /model 命令写入）
    3. AGENT_MODEL_PROVIDER 环境变量
    4. ProviderSpec 自动检测：codex → ark → openai → anthropic → deepseek → gateway → fake

    fake / fake_tool 模式不需要任何 API Key，用于调试。
    template_name: Prompt 模板名称（对应 prompts/ 目录下的 YAML 文件）；不传时读取 session metadata / 环境变量。
    """
    effective_session_id = session_id or os.getenv("AGENT_SESSION_ID") or f"debug-{uuid.uuid4()}"
    effective_cwd = Path(cwd or _WORKSPACE_ROOT).expanduser().resolve()
    session_info = None
    try:
        session_info = _session_store.get_session(effective_session_id)
    except Exception:
        session_info = None
    user_context = _user_store.get_user_context(
        user_id=user_id or (session_info or {}).get("user_id") or None,
        org_id=org_id or (session_info or {}).get("org_id") or None,
    )
    effective_user_id = user_context["user"]["id"]
    session_owner = str((session_info or {}).get("user_id") or "")
    if session_info is not None and user_id is not None and session_owner != str(effective_user_id):
        raise PermissionError("Session does not belong to current user")
    if session_owner and session_owner != str(effective_user_id):
        raise PermissionError("Session does not belong to current user")
    effective_org_id = str((session_info or {}).get("org_id") or user_context["organization"]["id"])
    if session_info is None:
        with contextlib.suppress(Exception):
            _session_store.create_session(
                session_id=effective_session_id,
                user_id=effective_user_id,
                org_id=effective_org_id,
                metadata={"user_id": effective_user_id, "org_id": effective_org_id},
            )
    selection = get_session_model_selection(effective_session_id)
    effective_template_name = get_session_prompt_template(effective_session_id, explicit_template_name=template_name)
    provider = (mode or selection["model_provider"] or "").lower()
    if not provider or provider == "auto":
        provider = _detect_provider_from_env()
    session_model_id = None if mode else selection.get("model_id")
    effective_model_id = _model_id_for_provider(provider, model_id=session_model_id)
    try:
        _billing_store.ensure_model(model_id=effective_model_id, provider=provider, display_name=effective_model_id)
    except Exception:
        pass
    session_memory_config = SessionMemoryConfig.from_env()
    session_memory = SessionMemoryManager(
        effective_session_id,
        config=session_memory_config,
        cwd=effective_cwd,
    ) if session_memory_config.enabled else None

    # ── 使用 PromptStore 创建 ContextBuilder ──
    try:
        context_builder = _prompt_store.create_context_builder(
            effective_template_name,
            session_id=effective_session_id,
            cwd=effective_cwd,
        )
    except FileNotFoundError:
        # 模板不存在时回退到硬编码
        context_builder = ContextBuilder(
            product_name="MyAgent",
            base_instructions=(
                "You are a backend-friendly general agent. "
                "Use tools when needed and produce concise, useful answers."
            ),
            dynamic_sections=[
                system_prompt_section("cwd", lambda: f"Working directory: {effective_cwd}"),
            ],
        )
        context_builder.append_system_context({"session_id": effective_session_id})

    # ── Skill discovery prompt：按 runtime cwd 隔离，避免并行工作区互相污染 ──
    runtime_skill_store = SkillStore(cwd=effective_cwd)
    context_builder.dynamic_sections.append(
        uncached_system_prompt_section("companion", lambda: build_companion_prompt(user_context))
    )
    context_builder.dynamic_sections.append(
        uncached_system_prompt_section("skills", lambda: build_skill_prompt_section(runtime_skill_store))
    )
    context_builder.dynamic_sections.append(
        uncached_system_prompt_section(
            "plan-mode",
            lambda: plan_mode_system_prompt(effective_session_id, cwd=effective_cwd),
        )
    )
    context_builder.dynamic_sections.append(
        uncached_system_prompt_section(
            "agent-team",
            lambda: _team_store.render_context(session_id=effective_session_id, user_id=effective_user_id),
        )
    )

    # fake 模式
    if provider == "fake_thinking":
        model = ScriptedModelClient(
            [
                ModelResponse(
                    content=[
                        ThinkingBlock(thinking="我会先确认请求，再给出简短响应。"),
                        TextBlock("这是 fake_thinking 模型响应。"),
                    ],
                    stop_reason="end_turn",
                )
            ]
        )
    elif provider == "fake_tool":
        model: ModelClient = ScriptedModelClient(
            [
                ModelResponse(
                    content=[
                        TextBlock("收到，我先用 echo 工具做一次链路验证。"),
                        ToolUseBlock(id="toolu_debug_1", name="echo", input={"text": "agent loop ok"}),
                    ],
                    stop_reason="tool_use",
                ),
                ModelResponse(content=[TextBlock("工具链路已跑通：agent loop ok")], stop_reason="end_turn"),
            ]
        )
    elif provider == "fake":
        model = ScriptedModelClient(
            [
                ModelResponse(
                    content=[TextBlock("这是 fake 模型响应。服务、AgentRuntime 和事件流已启动，可以开始调试。")],
                    stop_reason="end_turn",
                )
            ]
        )
    else:
        model = _create_model_from_env(provider, model_id=effective_model_id)

    vision_client = _create_vision_client_from_env(provider)

    def create_permission_policy() -> StaticPermissionPolicy:
        return StaticPermissionPolicy(
            allow={"echo", "TodoWrite", "Task", "Skill", "read_text_file", "list_directory", "grep", "glob", "bash_output", "UnderstandImage", "GenerateImage", "GenerateVideo", "VideoApi", "CreativeKBSearchV3", "TeamList", "ReadInbox", "SendMessage"},
            ask={"write_text_file", "edit_file", "bash", "kill_shell", "FeishuApi", "Agent", "TeamCreate", "TeamDelete"},
            session_id=effective_session_id,
            cwd=effective_cwd,
        )

    def create_sub_context_builder(child_session_id: str, subagent_type: str | None = None) -> ContextBuilder:
        try:
            builder = _prompt_store.create_context_builder(
                effective_template_name,
                session_id=child_session_id,
                cwd=effective_cwd,
            )
        except FileNotFoundError:
            builder = ContextBuilder(
                product_name="MyAgent Subagent",
                base_instructions=(
                    "You are a focused coding subagent. Complete the task you were given, "
                    "then return a concise final summary."
                ),
                dynamic_sections=[
                    system_prompt_section("cwd", lambda: f"Working directory: {effective_cwd}"),
                ],
            )
            builder.append_system_context({"session_id": child_session_id})
        builder.append_prompt = (
            "# Subagent Instructions\n"
            f"You are a {subagent_type or 'general-purpose'} subagent, not the main agent.\n"
            "Use your tools directly to complete the assigned task. Do not delegate further.\n"
            "Keep intermediate exploration out of the final response. Return only the conclusion, "
            "key evidence, files changed if any, and unresolved issues."
        )
        builder.dynamic_sections.append(
            uncached_system_prompt_section("skills", lambda: build_skill_prompt_section(runtime_skill_store))
        )
        return builder

    def create_sub_tools() -> ToolRegistry:
        return ToolRegistry([
            EchoTool(),
            SkillTool(runtime_skill_store),
            ReadTextFileTool(),
            WriteTextFileTool(),
            EditFileTool(),
            ListDirectoryTool(),
            GrepTool(),
            GlobTool(),
            WebFetchTool(),
            WebSearchTool(),
            FeishuApiTool(_feishu_token_store),
            UnderstandImageTool(_attachment_store, vision_client),
            GenerateImageTool(_attachment_store, get_image_generation_client),
            GenerateVideoTool(_attachment_store, get_video_generation_client),
            VideoApiTool(get_video_generation_client),
            CreativeKBSearchV3Tool(),
            SendMessageTool(_team_store),
            ReadInboxTool(_team_store),
            BashTool(),
            BashOutputTool(),
            KillShellTool(),
        ])

    tool_search = ToolSearchTool()
    tools = ToolRegistry([
        EchoTool(),
        tool_search,
        TodoWriteTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        TaskCreateTool(_task_store),
        TaskUpdateTool(_task_store),
        TaskListTool(_task_store),
        TaskGetTool(_task_store),
        TeamCreateTool(_team_store),
        TeamListTool(_team_store),
        TeamDeleteTool(_team_store),
        SkillTool(
            runtime_skill_store,
            model=model,
            tools_factory=create_sub_tools,
            context_builder_factory=create_sub_context_builder,
            permission_policy_factory=create_permission_policy,
        ),
        TaskTool(
            model=model,
            sub_tools_factory=create_sub_tools,
            context_builder_factory=create_sub_context_builder,
            permission_policy_factory=create_permission_policy,
        ),
        AgentTool(
            team_store=_team_store,
            model=model,
            sub_tools_factory=create_sub_tools,
            context_builder_factory=create_sub_context_builder,
            permission_policy_factory=create_permission_policy,
            session_store=_session_store,
        ),
        SendMessageTool(_team_store),
        ReadInboxTool(_team_store),
        ReadTextFileTool(),
        WriteTextFileTool(),
        EditFileTool(),
        ListDirectoryTool(),
        GrepTool(),
        GlobTool(),
        WebFetchTool(),
        WebSearchTool(),
        FeishuApiTool(_feishu_token_store),
        UnderstandImageTool(_attachment_store, vision_client),
        GenerateImageTool(_attachment_store, get_image_generation_client),
        GenerateVideoTool(_attachment_store, get_video_generation_client),
        VideoApiTool(get_video_generation_client),
        CreativeKBSearchV3Tool(),
        BashTool(),
        BashOutputTool(),
        KillShellTool(),
    ])
    tool_search.bind(tools)

    permission_policy = create_permission_policy()
    config = AgentRuntimeConfig(
        session_id=effective_session_id,
        skill_store=runtime_skill_store,
        session_memory_enabled=session_memory_config.enabled,
        model_id=effective_model_id,
        model_provider=provider,
        billing_store=_billing_store,
        user_id=effective_user_id,
        org_id=effective_org_id,
        quota_store=_quota_store,
        metadata={
            "user_id": effective_user_id,
            "org_id": effective_org_id,
            "account_uuid": user_context["user"].get("account_uuid"),
            "organization_uuid": user_context["organization"].get("organization_uuid"),
            "cwd": str(effective_cwd),
            "prompt_template": effective_template_name,
        },
        cwd=str(effective_cwd),
    )

    # ── 使用 HookStore 创建 HookEngine ──
    try:
        hook_engine = _hook_store.create_engine("default")
    except FileNotFoundError:
        from agent_core.hooks.engine import HookEngine
        hook_engine = HookEngine()

    return AgentRuntime(
        model=model,
        tools=tools,
        context_builder=context_builder,
        permission_policy=permission_policy,
        config=config,
        session_store=_session_store,
        ask_callback=ask_callback,
        hook_engine=hook_engine,
        session_memory=session_memory,
    )
