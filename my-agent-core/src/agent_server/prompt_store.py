"""Prompt 模板存储服务。

YAML 文件作为 source of truth，PromptStore 负责读写 YAML 并
将其转换为 ContextBuilder 实例。
"""
from __future__ import annotations

import os
import subprocess
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Awaitable

import yaml

from agent_core.context.builder import (
    ContextBuilder,
    SystemPromptSection,
    system_prompt_section,
    uncached_system_prompt_section,
)

# ── 动态 Context 计算 ────────────────────────────────────────

MAX_GIT_STATUS_CHARS = 2000


def _is_env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_git(cwd: Path, args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


@lru_cache(maxsize=64)
def _compute_git_status(cwd_value: str | None = None) -> str | None:
    """Return a cached git snapshot, mirroring Claude Code's conversation-start status."""
    if _is_env_truthy("AGENT_DISABLE_GIT_STATUS"):
        return None

    cwd = Path(cwd_value).expanduser().resolve() if cwd_value else Path.cwd()
    inside = _run_git(cwd, ["rev-parse", "--is-inside-work-tree"])
    if inside.lower() != "true":
        return None

    branch = _run_git(cwd, ["branch", "--show-current"]) or "(detached)"
    main_branch = _run_git(cwd, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"])
    if main_branch.startswith("origin/"):
        main_branch = main_branch.removeprefix("origin/")
    if not main_branch:
        main_branch = _run_git(cwd, ["config", "init.defaultBranch"]) or "main"
    status = _run_git(cwd, ["status", "--short"])
    log = _run_git(cwd, ["log", "--oneline", "-n", "5"])
    user_name = _run_git(cwd, ["config", "user.name"])

    truncated_status = status
    if len(truncated_status) > MAX_GIT_STATUS_CHARS:
        truncated_status = (
            truncated_status[:MAX_GIT_STATUS_CHARS]
            + '\n... (truncated because it exceeds 2k characters. If you need more information, run "git status" using BashTool)'
        )

    return "\n\n".join(
        [
            "This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.",
            f"Current branch: {branch}",
            f"Main branch (you will usually use this for PRs): {main_branch}",
            *([f"Git user: {user_name}"] if user_name else []),
            f"Status:\n{truncated_status or '(clean)'}",
            f"Recent commits:\n{log or '(none)'}",
        ]
    )


def _claude_md_search_roots(cwd_value: str | None = None) -> list[Path]:
    cwd = Path(cwd_value).expanduser().resolve() if cwd_value else Path.cwd().resolve()
    roots = [cwd, *cwd.parents]
    extra = os.getenv("AGENT_CLAUDE_MD_DIRS", "")
    for part in extra.split(os.pathsep):
        if part.strip():
            roots.append(Path(part).expanduser().resolve())
    return roots


@lru_cache(maxsize=64)
def _compute_claude_md(cwd_value: str | None = None) -> str | None:
    """Discover and concatenate project-level CLAUDE.md instruction files."""
    if _is_env_truthy("CLAUDE_CODE_DISABLE_CLAUDE_MDS") or _is_env_truthy("AGENT_DISABLE_CLAUDE_MDS"):
        return None

    files: list[Path] = []
    seen: set[Path] = set()
    for root in _claude_md_search_roots(cwd_value):
        candidate = root / "CLAUDE.md"
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        files.append(candidate)

    if not files:
        return None

    # Parent instructions first, then more specific cwd-level instructions.
    files.sort(key=lambda p: len(p.parts))
    rendered: list[str] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            rendered.append(f"# Project Instructions: {path}\n\n{content}")
    if not rendered:
        return None
    return "\n\n".join(rendered)


def clear_system_prompt_cache() -> None:
    """Clear cached dynamic prompt sections. Called by /clear and compact-style resets."""
    _compute_git_status.cache_clear()
    _compute_claude_md.cache_clear()


# ── 内置 Compute 函数注册表 ──────────────────────────────────

_COMPUTE_REGISTRY: dict[str, Callable[[], str | None]] = {
    "os_getcwd": lambda: f"Working directory: {os.getcwd()}",
    "current_date": lambda: f"Today's date is {date.today().isoformat()}.",
    "env_info": lambda: f"Environment: {os.name}, Python {os.sys.version_info.major}.{os.sys.version_info.minor}",
    "language": lambda: "Language: English",
    "git_status": _compute_git_status,
    "claude_md": _compute_claude_md,
}


def _compute_registry_for_cwd(cwd: str | Path | None = None) -> dict[str, Callable[[], str | None]]:
    resolved = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
    return {
        **_COMPUTE_REGISTRY,
        "os_getcwd": lambda: f"Working directory: {resolved}",
        "git_status": lambda: _compute_git_status(str(resolved)),
        "claude_md": lambda: _compute_claude_md(str(resolved)),
    }


def register_compute(name: str, fn: Callable[[], str | None]) -> None:
    """注册自定义 compute 函数到全局注册表。"""
    _COMPUTE_REGISTRY[name] = fn


# ── YAML Schema 验证 ─────────────────────────────────────────

_REQUIRED_METADATA = {"name", "description", "version"}

SOUL_DESIGN_PRINCIPLES: list[dict[str, str]] = [
    {
        "title": "Be genuinely helpful",
        "content": "Skip performative filler and solve the user's actual problem. Actions and evidence matter more than reassuring phrases.",
    },
    {
        "title": "Have useful opinions",
        "content": "Prefer clear judgment over bland neutrality. Disagree when the evidence supports it, and explain the tradeoff briefly.",
    },
    {
        "title": "Be resourceful before asking",
        "content": "Read available files, inspect context, search, and try focused diagnostics before asking the user. Ask only when the answer changes architecture, safety, or intent.",
    },
    {
        "title": "Earn trust through competence",
        "content": "Be careful with external or public actions, but be bold with local reversible work. Verify behavior whenever practical.",
    },
    {
        "title": "Respect privacy and voice",
        "content": "Private data stays private. In shared or messaging surfaces, do not speak as the user and never send half-baked replies.",
    },
    {
        "title": "Maintain continuity",
        "content": "Use durable project/session memory when provided. Treat the latest user request and current tool results as authoritative when they conflict with older context.",
    },
]


def _validate_template(data: dict[str, Any]) -> None:
    """验证 YAML 模板结构。"""
    if "metadata" not in data:
        raise ValueError("Template must have a 'metadata' section")
    for key in _REQUIRED_METADATA:
        if key not in data["metadata"]:
            raise ValueError(f"metadata must contain '{key}'")
    if "sections" not in data and "base_instructions" not in data:
        raise ValueError("Template must have 'sections' or 'base_instructions'")
    principles = data.get("design_principles")
    if principles is not None and not isinstance(principles, list):
        raise ValueError("design_principles must be a list")
    for item in principles or []:
        if not isinstance(item, dict):
            raise ValueError("each design principle must be a mapping")


# ── PromptStore ──────────────────────────────────────────────


class PromptStore:
    """YAML 文件驱动的 Prompt 模板存储。

    职责：
    - CRUD: 列出/读取/创建/更新/删除 YAML 模板文件
    - 转换: 将 YAML 模板转换为 ContextBuilder 实例
    - 预览: 渲染模板为完整 system prompt 文本
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        if prompts_dir is None:
            prompts_dir = Path(__file__).resolve().parent / "prompts"
        self._prompts_dir = prompts_dir
        self._prompts_dir.mkdir(parents=True, exist_ok=True)

    # ── CRUD ──

    def list_templates(self) -> list[dict[str, Any]]:
        """列出所有模板的元数据。"""
        templates = []
        for yaml_path in sorted(self._prompts_dir.glob("*.yaml")):
            try:
                data = self._load_yaml(yaml_path)
                templates.append(data.get("metadata", {"name": yaml_path.stem}))
            except Exception:
                templates.append({"name": yaml_path.stem, "error": "failed to parse"})
        return templates

    def get_template(self, name: str) -> dict[str, Any]:
        """获取指定模板的完整数据。"""
        path = self._resolve_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found")
        return self._load_yaml(path)

    def create_template(self, data: dict[str, Any]) -> dict[str, Any]:
        """创建新模板。"""
        _validate_template(data)
        name = data["metadata"]["name"]
        path = self._resolve_path(name)
        if path.exists():
            raise FileExistsError(f"Template '{name}' already exists")
        self._save_yaml(path, data)
        return data

    def update_template(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """更新已有模板。"""
        _validate_template(data)
        path = self._resolve_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found")
        new_name = data["metadata"]["name"]
        if new_name != name:
            path.unlink()
        self._save_yaml(self._resolve_path(new_name), data)
        return data

    def delete_template(self, name: str) -> None:
        """删除模板。"""
        path = self._resolve_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Template '{name}' not found")
        path.unlink()

    # ── 转换为 ContextBuilder ──

    def create_context_builder(
        self,
        name: str,
        *,
        session_id: str | None = None,
        override_prompt: str | None = None,
        agent_prompt: str | None = None,
        custom_prompt: str | None = None,
        append_prompt: str | None = None,
        cwd: str | Path | None = None,
    ) -> ContextBuilder:
        """从 YAML 模板创建 ContextBuilder 实例。

        这是 PromptStore 的核心方法：YAML → ContextBuilder。
        """
        data = self.get_template(name)

        # 构建 static_sections
        static_sections = self._render_static_sections(data)

        # 构建 dynamic_sections
        dynamic_sections = self._render_dynamic_sections(data, cwd=cwd)

        # 构建 context 注入
        prepend_ctx, append_ctx = self._render_context(data, session_id=session_id)

        # 优先级链参数
        effective_override = override_prompt or data.get("override_prompt")
        effective_agent = agent_prompt or data.get("agent_prompt")
        effective_custom = custom_prompt or data.get("custom_prompt")
        effective_append = append_prompt or data.get("append_prompt")

        builder = ContextBuilder(
            product_name=data.get("product_name", "MyAgent"),
            base_instructions=data.get("base_instructions", ""),
            static_sections=static_sections,
            override_prompt=effective_override,
            agent_prompt=effective_agent,
            custom_prompt=effective_custom,
            append_prompt=effective_append,
            dynamic_sections=dynamic_sections,
        )

        if prepend_ctx:
            builder.prepend_user_context(prepend_ctx)
        if append_ctx:
            builder.append_system_context(append_ctx)

        return builder

    # ── 预览渲染 ──

    async def preview(self, name: str, *, session_id: str | None = None) -> str:
        """预览模板渲染后的完整 system prompt 文本。"""
        builder = self.create_context_builder(name, session_id=session_id)
        from agent_core.types import system_prompt_to_str
        prompt = await builder.build()
        return system_prompt_to_str(prompt)

    # ── 内部方法 ──

    def _resolve_path(self, name: str) -> Path:
        """将模板名解析为文件路径（防止路径遍历）。"""
        safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
        if safe_name != name:
            raise ValueError(f"Invalid template name: {name}")
        return self._prompts_dir / f"{safe_name}.yaml"

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """加载 YAML 文件。"""
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Template must be a YAML mapping, got {type(data).__name__}")
        return data

    def _save_yaml(self, path: Path, data: dict[str, Any]) -> None:
        """保存 YAML 文件。"""
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _render_static_sections(self, data: dict[str, Any]) -> list[str]:
        """将 YAML sections 渲染为字符串列表。"""
        sections: list[str] = []

        product_name = data.get("product_name", "MyAgent")
        sections.append(f"# {product_name}")

        base_instructions = data.get("base_instructions", "")
        if base_instructions:
            sections.append(base_instructions.strip())

        principles = data.get("design_principles") or []
        if principles:
            rendered_principles: list[str] = []
            for item in principles:
                if not isinstance(item, dict) or item.get("enabled") is False:
                    continue
                title = str(item.get("title") or item.get("name") or "").strip()
                content = str(item.get("content") or item.get("text") or "").strip()
                if title and content:
                    rendered_principles.append(f"- **{title}:** {content}")
                elif content:
                    rendered_principles.append(f"- {content}")
                elif title:
                    rendered_principles.append(f"- **{title}**")
            if rendered_principles:
                sections.append("## Design Principles\n\n" + "\n".join(rendered_principles))

        for sec in data.get("sections", []):
            title = sec.get("title", "")
            content = sec.get("content", "")
            if title and content:
                sections.append(f"## {title}\n\n{content.strip()}")
            elif content:
                sections.append(content.strip())
            elif title:
                sections.append(f"## {title}")

        return [s for s in sections if s.strip()]

    def _render_dynamic_sections(self, data: dict[str, Any], *, cwd: str | Path | None = None) -> list[SystemPromptSection]:
        """将 YAML dynamic_sections 定义转换为 SystemPromptSection 列表。"""
        registry = _compute_registry_for_cwd(cwd)
        result: list[SystemPromptSection] = []
        for dyn in data.get("dynamic_sections", []):
            name = dyn["name"]
            compute_name = dyn.get("compute", "")
            cache_break = dyn.get("cache_break", False)

            fn = registry.get(compute_name)
            if fn is None:
                continue

            if cache_break:
                result.append(uncached_system_prompt_section(name, fn, reason=dyn.get("reason", "")))
            else:
                result.append(system_prompt_section(name, fn))

        return result

    def _render_context(
        self, data: dict[str, Any], *, session_id: str | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        """渲染上下文注入点。"""
        ctx_config = data.get("context", {})

        def _resolve_values(mapping: dict[str, str]) -> dict[str, str]:
            resolved = {}
            for k, v in mapping.items():
                if v == "auto" and k == "session_id":
                    resolved[k] = session_id or os.getenv("AGENT_SESSION_ID") or "debug-session"
                else:
                    resolved[k] = str(v)
            return resolved

        prepend = _resolve_values(ctx_config.get("prepend", {}))
        append = _resolve_values(ctx_config.get("append", {}))
        return prepend, append
