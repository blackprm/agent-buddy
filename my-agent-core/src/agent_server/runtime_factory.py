from __future__ import annotations

import os
import uuid
import contextlib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_core.billing.store import BillingStore
from agent_core.buddy import build_companion_prompt
from agent_core.context.builder import ContextBuilder, system_prompt_section, uncached_system_prompt_section
from agent_core.core.agent import AgentRuntime, AgentRuntimeConfig
from agent_core.hooks.store import HookStore
from agent_core.integrations.feishu import FeishuApiTool, FeishuTokenStore
from agent_core.memory.session_memory import SessionMemoryConfig, SessionMemoryManager
from agent_core.model.base import ModelClient, ModelResponse
from agent_core.model.fake import ScriptedModelClient
from agent_core.permissions.policy import StaticPermissionPolicy
from agent_core.quota.store import QuotaStore
from agent_core.skills.store import SkillStore, build_skill_prompt_section
from agent_core.session.store import create_session_store, SessionStore
from agent_core.tasks import FileTaskStore, TaskCreateTool, TaskGetTool, TaskListTool, TaskUpdateTool
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
from agent_core.tools.skill import SkillTool
from agent_core.tools.subagent import TaskTool
from agent_core.tools.web import WebFetchTool, WebSearchTool
from agent_core.types import TextBlock, ThinkingBlock, ToolUseBlock
from agent_core.users.store import UserStore
from agent_core.worktree import WorktreeManager
from agent_core.plan_mode import plan_mode_system_prompt
from agent_server.prompt_store import PromptStore

# 自动加载 .env — 从项目根目录（src 的父目录）查找
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()
_env_path = _PROJECT_ROOT / ".env"
load_dotenv(_env_path, override=True)
_task_store = FileTaskStore(task_list_id="tasklist")


def get_task_store() -> FileTaskStore:
    return _task_store


def create_worktree_manager(*, cwd: str | Path | None = None) -> WorktreeManager:
    try:
        hook_engine = _hook_store.create_engine("default")
    except FileNotFoundError:
        from agent_core.hooks.engine import HookEngine
        hook_engine = HookEngine()
    return WorktreeManager(cwd=cwd or _WORKSPACE_ROOT, hook_engine=hook_engine)


# ── 模型提供商注册表 ──────────────────────────────────────────

_PROVIDERS: dict[str, type[ModelClient]] = {}


def register_provider(name: str, cls: type[ModelClient]) -> None:
    """注册模型提供商。扩展时调用此函数即可。"""
    _PROVIDERS[name.lower()] = cls


def _init_providers() -> None:
    """延迟注册，避免 import 未安装的 SDK。"""
    if _PROVIDERS:
        return
    try:
        from agent_core.model.anthropic_adapter import AnthropicMessagesModelClient
        register_provider("anthropic", AnthropicMessagesModelClient)
    except ImportError:
        pass
    try:
        from agent_core.model.openai_adapter import OpenAICompatibleModelClient
        register_provider("openai", OpenAICompatibleModelClient)
        # 火山方舟 / ARK 也走 OpenAI 兼容协议
        register_provider("ark", OpenAICompatibleModelClient)
        register_provider("volcengine", OpenAICompatibleModelClient)
        register_provider("doubao", OpenAICompatibleModelClient)
        # DeepSeek 等也兼容
        register_provider("deepseek", OpenAICompatibleModelClient)
        # 自定义 OpenAI-compatible 网关
        register_provider("gateway", OpenAICompatibleModelClient)
    except ImportError:
        pass


# ── 从环境变量创建 ModelClient ────────────────────────────────


def _create_model_from_env(provider: str, *, model_id: str | None = None) -> ModelClient:
    """根据 provider 名称从环境变量创建 ModelClient。"""
    _init_providers()

    cls = _PROVIDERS.get(provider)
    if cls is None:
        available = ", ".join(sorted(_PROVIDERS.keys())) or "none (install anthropic or openai SDK)"
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available}. "
            f"Set AGENT_MODEL_PROVIDER to one of these."
        )

    # Anthropic 系列
    if provider == "anthropic":
        return cls(
            model=model_id or os.getenv("ANTHROPIC_MODEL") or os.getenv("MODEL_ID") or "claude-3-5-sonnet-latest",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
        )

    # OpenAI 兼容系列（openai / ark / volcengine / doubao / deepseek / gateway）
    if provider in ("openai", "ark", "volcengine", "doubao", "deepseek", "gateway"):
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or ""
            base_url = os.getenv("OPENAI_BASE_URL") or None
            model = model_id or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID") or "gpt-4o"
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY") or ""
            base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
            model = model_id or os.getenv("DEEPSEEK_MODEL") or os.getenv("MODEL_ID") or "deepseek-chat"
        elif provider == "gateway":
            api_key = os.getenv("GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = os.getenv("GATEWAY_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
            model = model_id or os.getenv("GATEWAY_MODEL") or os.getenv("MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4o"
        else:
            api_key = os.getenv("ARK_API_KEY") or ""
            base_url = os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
            model = model_id or os.getenv("ARK_MODEL") or os.getenv("MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4o"
        return cls(model=model, api_key=api_key, base_url=base_url)

    # 兜底：尝试通用构造
    return cls(
        model=model_id or os.getenv("MODEL_ID") or "unknown",
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"),
    )


def _model_id_for_provider(provider: str, *, model_id: str | None = None) -> str:
    """Resolve the model id with the same precedence used to create clients."""
    if model_id:
        return model_id
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL") or os.getenv("MODEL_ID") or "claude-3-5-sonnet-latest"
    if provider == "openai":
        return os.getenv("OPENAI_MODEL") or os.getenv("MODEL_ID") or "gpt-4o"
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL") or os.getenv("MODEL_ID") or "deepseek-chat"
    if provider == "gateway":
        return os.getenv("GATEWAY_MODEL") or os.getenv("MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4o"
    if provider in ("ark", "volcengine", "doubao"):
        return os.getenv("ARK_MODEL") or os.getenv("MODEL_ID") or os.getenv("OPENAI_MODEL") or "gpt-4o"
    if provider in ("fake", "fake_tool", "fake_thinking"):
        return provider
    return os.getenv("MODEL_ID") or provider or "unknown"


def _detect_provider_from_env() -> str:
    provider = (os.getenv("AGENT_MODEL_PROVIDER") or "").lower()
    if provider and provider != "auto":
        return provider
    if os.getenv("ARK_API_KEY"):
        return "ark"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "fake"


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
    return {
        "model_provider": provider,
        "model_id": _model_id_for_provider(provider, model_id=model_id or None),
    }


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
    template_name: str = "default",
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
    4. 自动检测：有 ARK_API_KEY → ark，有 ANTHROPIC_API_KEY → anthropic，否则 → fake

    fake / fake_tool 模式不需要任何 API Key，用于调试。
    template_name: Prompt 模板名称（对应 prompts/ 目录下的 YAML 文件）
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
            template_name,
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

    def create_permission_policy() -> StaticPermissionPolicy:
        return StaticPermissionPolicy(
            allow={"echo", "TodoWrite", "Task", "Skill", "read_text_file", "list_directory", "grep", "glob", "bash_output"},
            ask={"write_text_file", "edit_file", "bash", "kill_shell", "FeishuApi"},
            session_id=effective_session_id,
            cwd=effective_cwd,
        )

    def create_sub_context_builder(child_session_id: str, subagent_type: str | None = None) -> ContextBuilder:
        try:
            builder = _prompt_store.create_context_builder(
                template_name,
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
        ReadTextFileTool(),
        WriteTextFileTool(),
        EditFileTool(),
        ListDirectoryTool(),
        GrepTool(),
        GlobTool(),
        WebFetchTool(),
        WebSearchTool(),
        FeishuApiTool(_feishu_token_store),
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
