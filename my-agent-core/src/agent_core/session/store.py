"""会话持久化 — Protocol + 序列化 + 工厂注册表。

开闭原则：
- SessionStore Protocol：稳定的抽象接口，不随实现变化
- 注册表 + 工厂：新增实现只需 register_session_store("mysql", MySQLStore)，
  不修改任何已有代码
- 配置驱动：SESSION_STORE_TYPE 环境变量选择实现
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from agent_core.types import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


# ── 序列化 / 反序列化 ────────────────────────────────────────


def serialize_block(block: ContentBlock) -> dict[str, Any]:
    """将 ContentBlock 序列化为 JSON-safe dict。"""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    elif isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    elif isinstance(block, ToolResultBlock):
        return {"type": "tool_result", "tool_use_id": block.tool_use_id, "content": block.content, "is_error": block.is_error}
    return {"type": "unknown", "data": str(block)}


def deserialize_block(d: dict[str, Any]) -> ContentBlock:
    """将 dict 反序列化为 ContentBlock。"""
    t = d.get("type")
    if t == "text":
        return TextBlock(text=d["text"])
    elif t == "thinking":
        return ThinkingBlock(thinking=d["thinking"])
    elif t == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], input=d.get("input", {}))
    elif t == "tool_result":
        return ToolResultBlock(tool_use_id=d["tool_use_id"], content=d["content"], is_error=d.get("is_error", False))
    return TextBlock(text=json.dumps(d, ensure_ascii=False))


def serialize_message(msg: Message) -> str:
    """序列化 Message 为 JSON 字符串。"""
    blocks = [serialize_block(b) for b in msg.content]
    return json.dumps({"role": msg.role, "content": blocks, "metadata": msg.metadata}, ensure_ascii=False)


def deserialize_message(data: str) -> Message:
    """反序列化 JSON 字符串为 Message。"""
    d = json.loads(data)
    blocks = [deserialize_block(b) for b in d.get("content", [])]
    return Message(role=d["role"], content=blocks, metadata=d.get("metadata", {}))


# ── SessionStore Protocol ─────────────────────────────────────


@runtime_checkable
class SessionStore(Protocol):
    """会话持久化接口 — 对扩展开放，对修改关闭。

    新增实现（MySQL、PostgreSQL、Redis...）只需实现此 Protocol，
    然后通过 register_session_store() 注册即可。
    """

    def create_session(
        self,
        *,
        session_id: str | None = None,
        metadata: dict | None = None,
        user_id: str = "",
        org_id: str = "",
    ) -> str:
        """创建新会话，返回 session_id。"""
        ...

    def save_message(self, session_id: str, message: Message, turn: int = 0) -> None:
        """保存单条消息。"""
        ...

    def save_messages(self, session_id: str, messages: list[Message], start_turn: int = 0) -> None:
        """批量保存消息（覆盖 start_turn 及之后的消息）。"""
        ...

    def load_messages(self, session_id: str) -> list[Message]:
        """加载会话的全部消息。"""
        ...

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话元信息。"""
        ...

    def list_sessions(self, *, limit: int = 50, offset: int = 0, user_id: str | None = None, org_id: str | None = None) -> list[dict[str, Any]]:
        """列出会话（按最近活跃排序），可按用户/组织过滤。"""
        ...

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其消息。返回是否成功。"""
        ...

    def update_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """更新会话元信息。"""
        ...


# ── 工厂注册表 ────────────────────────────────────────────────

_STORE_REGISTRY: dict[str, type] = {}
"""实现注册表。key = store 类型名，value = 实现类。"""


def register_session_store(name: str, cls: type) -> None:
    """注册 SessionStore 实现。

    用法：
        register_session_store("mysql", MySQLSessionStore)
    之后可通过 create_session_store("mysql", ...) 创建实例。
    """
    _STORE_REGISTRY[name.lower()] = cls


def create_session_store(store_type: str = "sqlite", **kwargs: Any) -> SessionStore:
    """工厂方法：根据类型名创建 SessionStore 实例。

    参数:
        store_type: 注册的类型名（"sqlite", "mysql", ...）
        **kwargs: 传递给实现类构造函数的参数

    环境变量:
        SESSION_STORE_TYPE: 覆盖 store_type 参数
        SESSION_STORE_URL: 连接字符串（如 mysql://...），传给实现类
    """
    import os

    # 环境变量覆盖
    effective_type = os.getenv("SESSION_STORE_TYPE", store_type).lower()

    # 延迟注册内置实现
    _ensure_builtin_registered()

    cls = _STORE_REGISTRY.get(effective_type)
    if cls is None:
        available = ", ".join(sorted(_STORE_REGISTRY.keys())) or "none"
        raise ValueError(
            f"Unknown session store type '{effective_type}'. Available: {available}. "
            f"Set SESSION_STORE_TYPE or register via register_session_store()."
        )

    # 如果传了 url 参数或环境变量，注入到 kwargs
    url = kwargs.pop("url", None) or os.getenv("SESSION_STORE_URL")
    if url:
        kwargs["url"] = url

    return cls(**kwargs)


_builtin_registered = False


def _ensure_builtin_registered() -> None:
    """延迟注册内置实现，避免循环 import。"""
    global _builtin_registered
    if _builtin_registered:
        return
    _builtin_registered = True

    # SQLite 是内置的，始终可用
    from agent_core.session.sqlite_store import SQLiteSessionStore
    register_session_store("sqlite", SQLiteSessionStore)
