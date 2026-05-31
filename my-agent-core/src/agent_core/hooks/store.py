"""HookStore — YAML 格式的 hooks 配置管理。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agent_core.hooks.types import HookEvent, HookMatcher, HookDefinition

logger = logging.getLogger(__name__)

_HOOKS_DIR = Path(__file__).resolve().parent


class HookStore:
    """Hooks 配置 CRUD 服务。

    YAML 格式:
        hooks:
          PreToolUse:
            - matcher: "Write|Edit"
              hooks:
                - type: command
                  command: "lint.sh"
                  timeout: 30
    """

    def __init__(self, hooks_dir: Path | str | None = None) -> None:
        self._hooks_dir = Path(hooks_dir) if hooks_dir else _HOOKS_DIR

    # ── 模板列表 ─────────────────────────────────────────

    def list_templates(self) -> list[str]:
        """列出所有 YAML 模板名。"""
        templates = []
        for p in sorted(self._hooks_dir.glob("*.yaml")):
            templates.append(p.stem)
        return templates

    # ── 加载 ─────────────────────────────────────────────

    def load(self, name: str = "default") -> list[HookMatcher]:
        """加载 YAML 配置，返回 HookMatcher 列表。"""
        path = self._resolve(name)
        if not path.exists():
            raise FileNotFoundError(f"Hooks template not found: {name}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return self._parse_matchers(data)

    def _parse_matchers(self, data: dict[str, Any]) -> list[HookMatcher]:
        """解析 YAML 数据为 HookMatcher 列表。"""
        matchers: list[HookMatcher] = []
        hooks_data = data.get("hooks", {})

        for event_name, entries in hooks_data.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                logger.warning("Unknown hook event: %s, skipping", event_name)
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                if isinstance(entry, dict):
                    matchers.append(HookMatcher.from_dict(event, entry))

        return matchers

    # ── 保存 ─────────────────────────────────────────────

    def save(self, name: str, matchers: list[HookMatcher]) -> None:
        """保存 HookMatcher 列表到 YAML。"""
        path = self._resolve(name)
        data = self._serialize_matchers(matchers)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _serialize_matchers(self, matchers: list[HookMatcher]) -> dict[str, Any]:
        """序列化 HookMatcher 列表为 YAML 结构。"""
        hooks: dict[str, list[dict[str, Any]]] = {}
        for event in HookEvent:
            hooks[event.value] = []

        for matcher in matchers:
            event_key = matcher.event.value
            hooks.setdefault(event_key, []).append(matcher.to_dict())

        return {"hooks": hooks}

    # ── CRUD ─────────────────────────────────────────────

    def get_raw(self, name: str = "default") -> dict[str, Any]:
        """获取原始 YAML 数据。"""
        path = self._resolve(name)
        if not path.exists():
            raise FileNotFoundError(f"Hooks template not found: {name}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def update_raw(self, name: str, data: dict[str, Any]) -> None:
        """更新原始 YAML 数据。"""
        path = self._resolve(name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def delete_template(self, name: str) -> None:
        """删除模板文件。"""
        if name == "default":
            raise ValueError("Cannot delete default template")
        path = self._resolve(name)
        if path.exists():
            path.unlink()

    def create_template(self, name: str, data: dict[str, Any] | None = None) -> None:
        """创建新模板。"""
        path = self._resolve(name)
        if path.exists():
            raise FileExistsError(f"Template already exists: {name}")
        template = data or {"hooks": {e.value: [] for e in HookEvent}}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # ── 创建 HookEngine ──────────────────────────────────

    def create_engine(self, name: str = "default") -> "HookEngine":
        """从 YAML 配置创建 HookEngine。"""
        from agent_core.hooks.engine import HookEngine

        matchers = self.load(name)
        return HookEngine(matchers=matchers)

    # ── 内部 ─────────────────────────────────────────────

    def _resolve(self, name: str) -> Path:
        """解析模板路径（防路径遍历）。"""
        safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._hooks_dir / f"{safe_name}.yaml"
