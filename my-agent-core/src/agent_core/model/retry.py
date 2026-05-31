from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_core.model.base import ModelClient, StreamDelta
from agent_core.types import Message, SystemPrompt


@dataclass(slots=True)
class ModelRetryConfig:
    max_retries: int = 2
    base_delay_ms: int = 250
    max_delay_ms: int = 4_000
    jitter: float = 0.25


@dataclass(slots=True)
class ModelRetryEvent:
    attempt: int
    max_retries: int
    retry_delay_ms: int
    error: str
    category: str
    reset_stream: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "error": self.error,
            "category": self.category,
            "reset_stream": self.reset_stream,
            "metadata": self.metadata,
        }


def categorize_model_error(exc: BaseException) -> tuple[str, bool]:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(token in text for token in ("prompt too long", "context_length", "context window", "maximum context", "413")):
        return "prompt_too_long", False
    if any(token in text for token in ("max output", "max_tokens", "output token")):
        return "max_output_tokens", False
    if any(token in text for token in ("401", "403", "auth", "api key", "permission")):
        return "authentication_failed", False
    if any(token in text for token in ("429", "rate limit", "529", "overload", "timeout", "temporarily", "connection", "econnreset", "server error", "5")):
        return "transient", True
    return "unknown", True


def _delay_ms(attempt: int, config: ModelRetryConfig) -> int:
    raw = min(config.max_delay_ms, config.base_delay_ms * (2 ** max(0, attempt - 1)))
    if config.jitter <= 0:
        return raw
    spread = int(raw * config.jitter)
    return max(0, raw + random.randint(-spread, spread))


async def stream_with_retries(
    model: ModelClient,
    *,
    system: str | SystemPrompt,
    messages: list[Message],
    tools: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    config: ModelRetryConfig | None = None,
) -> AsyncIterator[StreamDelta | ModelRetryEvent]:
    config = config or ModelRetryConfig()
    for attempt in range(1, config.max_retries + 2):
        try:
            async for delta in model.stream(system=system, messages=messages, tools=tools, metadata=metadata):
                yield delta
            return
        except Exception as exc:  # noqa: BLE001 - model boundary converts transient errors into retry events
            category, retryable = categorize_model_error(exc)
            if not retryable or attempt > config.max_retries:
                raise
            delay = _delay_ms(attempt, config)
            yield ModelRetryEvent(
                attempt=attempt,
                max_retries=config.max_retries,
                retry_delay_ms=delay,
                error=str(exc),
                category=category,
            )
            await asyncio.sleep(delay / 1000)
