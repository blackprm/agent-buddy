from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from agent_core.model.base import ModelClient
from agent_server.codex_credentials import codex_base_url, codex_model, resolve_codex_provider_credentials


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """声明式模型 provider 配置。

    新增 OpenAI-compatible 网关时只需要补一条 spec，runtime、slash command、
    terminal/admin 配置展示都复用同一套解析逻辑，避免在多个文件继续追加
    provider 分支。
    """

    name: str
    client_path: str | None
    default_model: str
    api_key_env: tuple[str, ...] = ()
    base_url_env: tuple[str, ...] = ()
    model_env: tuple[str, ...] = ()
    detect_env: tuple[str, ...] = ()
    default_base_url: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProviderConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    spec: ProviderSpec


OPENAI_COMPATIBLE_CLIENT = "agent_core.model.openai_adapter:OpenAICompatibleModelClient"
ANTHROPIC_CLIENT = "agent_core.model.anthropic_adapter:AnthropicMessagesModelClient"


_PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "codex": ProviderSpec(
        name="codex",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-5.4",
        api_key_env=("CODEX_API_KEY", "GATEWAY_API_KEY", "OPENAI_API_KEY"),
        base_url_env=("CODEX_BASE_URL",),
        model_env=("CODEX_MODEL", "MODEL_ID"),
        detect_env=("CODEX_API_KEY", "CODEX_BASE_URL"),
        default_base_url="http://localhost:8080/v1",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        client_path=ANTHROPIC_CLIENT,
        default_model="claude-3-5-sonnet-latest",
        api_key_env=("ANTHROPIC_API_KEY",),
        base_url_env=("ANTHROPIC_BASE_URL",),
        model_env=("ANTHROPIC_MODEL", "MODEL_ID"),
        detect_env=("ANTHROPIC_API_KEY",),
    ),
    "openai": ProviderSpec(
        name="openai",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-4o",
        api_key_env=("OPENAI_API_KEY",),
        base_url_env=("OPENAI_BASE_URL",),
        model_env=("OPENAI_MODEL", "MODEL_ID"),
        detect_env=("OPENAI_API_KEY",),
    ),
    "ark": ProviderSpec(
        name="ark",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-4o",
        api_key_env=("ARK_API_KEY",),
        base_url_env=("ARK_BASE_URL",),
        model_env=("ARK_MODEL", "MODEL_ID", "OPENAI_MODEL"),
        detect_env=("ARK_API_KEY",),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    "volcengine": ProviderSpec(
        name="volcengine",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-4o",
        api_key_env=("ARK_API_KEY",),
        base_url_env=("ARK_BASE_URL",),
        model_env=("ARK_MODEL", "MODEL_ID", "OPENAI_MODEL"),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    "doubao": ProviderSpec(
        name="doubao",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-4o",
        api_key_env=("ARK_API_KEY",),
        base_url_env=("ARK_BASE_URL",),
        model_env=("ARK_MODEL", "MODEL_ID", "OPENAI_MODEL"),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="deepseek-chat",
        api_key_env=("DEEPSEEK_API_KEY",),
        base_url_env=("DEEPSEEK_BASE_URL",),
        model_env=("DEEPSEEK_MODEL", "MODEL_ID"),
        detect_env=("DEEPSEEK_API_KEY",),
        default_base_url="https://api.deepseek.com",
    ),
    "gateway": ProviderSpec(
        name="gateway",
        client_path=OPENAI_COMPATIBLE_CLIENT,
        default_model="gpt-4o",
        api_key_env=("GATEWAY_API_KEY", "OPENAI_API_KEY"),
        base_url_env=("GATEWAY_BASE_URL", "OPENAI_BASE_URL"),
        model_env=("GATEWAY_MODEL", "MODEL_ID", "OPENAI_MODEL"),
        detect_env=("GATEWAY_API_KEY", "GATEWAY_BASE_URL"),
    ),
    "fake": ProviderSpec(name="fake", client_path=None, default_model="fake"),
    "fake_tool": ProviderSpec(name="fake_tool", client_path=None, default_model="fake_tool"),
    "fake_thinking": ProviderSpec(name="fake_thinking", client_path=None, default_model="fake_thinking"),
}

_DETECT_PRIORITY = ("ark", "openai", "anthropic", "deepseek", "gateway")


def provider_spec(name: str) -> ProviderSpec | None:
    return _PROVIDER_SPECS.get((name or "").lower())


def available_provider_names() -> set[str]:
    return set(_PROVIDER_SPECS)


def configured_provider_names() -> set[str]:
    configured: set[str] = set()
    if resolve_codex_provider_credentials():
        configured.add("codex")
    for name in _DETECT_PRIORITY:
        spec = _PROVIDER_SPECS[name]
        if any(_env_value(env_name) for env_name in spec.detect_env):
            configured.add(name)
    if not configured:
        configured.add("fake")
    return configured


def detect_provider_from_env() -> str:
    explicit = (os.getenv("AGENT_MODEL_PROVIDER") or "").strip().lower()
    if explicit and explicit != "auto":
        return explicit
    for name in _DETECT_PRIORITY:
        spec = _PROVIDER_SPECS[name]
        if any(_env_value(env_name) for env_name in spec.detect_env):
            return name
    return "fake"


def default_model_for_provider(provider: str, *, model_id: str | None = None) -> str:
    if model_id:
        return model_id
    if (provider or "").lower() == "codex":
        return codex_model(_PROVIDER_SPECS["codex"].default_model)
    spec = provider_spec(provider)
    if spec is None:
        return _env_value("MODEL_ID") or provider or "unknown"
    return _first_env(spec.model_env) or spec.default_model


def resolve_provider_config(provider: str, *, model_id: str | None = None) -> ResolvedProviderConfig:
    normalized = (provider or "").lower()
    spec = provider_spec(normalized)
    if spec is None:
        available = ", ".join(sorted(available_provider_names()))
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available}. "
            f"Set AGENT_MODEL_PROVIDER to one of these."
        )
    codex_credentials = resolve_codex_provider_credentials() if spec.name == "codex" else None
    return ResolvedProviderConfig(
        provider=spec.name,
        model=default_model_for_provider(spec.name, model_id=model_id),
        api_key=codex_credentials.api_key if codex_credentials else _first_env(spec.api_key_env),
        base_url=codex_base_url(spec.default_base_url or "http://localhost:8080/v1") if spec.name == "codex" else _first_env(spec.base_url_env) or spec.default_base_url,
        spec=spec,
    )


def create_model_from_env(provider: str, *, model_id: str | None = None) -> ModelClient:
    config = resolve_provider_config(provider, model_id=model_id)
    if config.spec.client_path is None:
        raise ValueError(f"Provider '{provider}' does not create a remote model client")
    cls = _load_client(config.spec.client_path)
    return cls(model=config.model, api_key=config.api_key or "", base_url=config.base_url)


def env_model_selection() -> dict[str, str]:
    provider = detect_provider_from_env()
    return {"model_provider": provider, "model_id": default_model_for_provider(provider)}


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _env_value(name)
        if value:
            return value
    return None


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _load_client(path: str) -> type[ModelClient]:
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Invalid provider client path: {path}")
    module = importlib.import_module(module_name)
    cls: Any = getattr(module, attr)
    return cls
