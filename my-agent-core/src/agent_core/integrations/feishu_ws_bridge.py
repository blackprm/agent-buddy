from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_core.core.agent import AgentRuntime
from agent_core.integrations.feishu import FeishuApiTool, FeishuTokenStore
from agent_core.tools.base import ToolContext


RuntimeFactory = Callable[..., AgentRuntime]

_FEISHU_DOMAIN = "https://open.feishu.cn"
_PROCESSED_TTL_SECONDS = 24 * 60 * 60
_MAX_PROCESSED_MESSAGES = 4096
_MAX_DIAGNOSTIC_LOGS = 200

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FeishuInboundMessage:
    message_id: str
    chat_id: str
    chat_type: str
    message_type: str
    text: str
    sender_type: str = ""
    sender_open_id: str = ""
    sender_user_id: str = ""
    root_id: str = ""
    thread_id: str = ""
    mentions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class FeishuBridgeStatus:
    user_id: str
    org_id: str
    state: str = "stopped"
    running: bool = False
    started_at: float | None = None
    stopped_at: float | None = None
    last_event_at: float | None = None
    last_reply_at: float | None = None
    last_error: str = ""
    last_message_id: str = ""
    last_session_id: str = ""
    last_ignored_reason: str = ""
    message_count: int = 0
    reply_count: int = 0
    ignored_count: int = 0
    bot_open_id: str = ""
    bot_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "state": self.state,
            "running": self.running,
            "started_at": self.started_at,
            "started_at_iso": _iso_from_ts(self.started_at),
            "stopped_at": self.stopped_at,
            "stopped_at_iso": _iso_from_ts(self.stopped_at),
            "last_event_at": self.last_event_at,
            "last_event_at_iso": _iso_from_ts(self.last_event_at),
            "last_reply_at": self.last_reply_at,
            "last_reply_at_iso": _iso_from_ts(self.last_reply_at),
            "last_error": self.last_error,
            "last_message_id": self.last_message_id,
            "last_session_id": self.last_session_id,
            "last_ignored_reason": self.last_ignored_reason,
            "message_count": self.message_count,
            "reply_count": self.reply_count,
            "ignored_count": self.ignored_count,
            "bot_open_id": self.bot_open_id,
            "bot_name": self.bot_name,
        }


class FeishuWebSocketBridge:
    """OpenClaw-style Feishu WebSocket ingress bridge for one MyAgent user.

    The Feishu SDK invokes event handlers synchronously before ACKing a frame.
    Keep that handler tiny: it only enqueues work onto a private asyncio loop,
    then returns immediately so Feishu does not wait for an LLM turn.
    """

    def __init__(
        self,
        *,
        token_store: FeishuTokenStore,
        runtime_factory: RuntimeFactory,
        api_tool_factory: Callable[[], FeishuApiTool] | None = None,
        domain: str = _FEISHU_DOMAIN,
    ) -> None:
        self._token_store = token_store
        self._runtime_factory = runtime_factory
        self._api_tool_factory = api_tool_factory or (lambda: FeishuApiTool(token_store))
        self._domain = domain.rstrip("/")
        self._lock = threading.RLock()
        self._handles: dict[tuple[str, str], _BridgeHandle] = {}

    def start(self, *, user_id: str, org_id: str) -> dict[str, Any]:
        key = _key(user_id, org_id)
        with self._lock:
            existing = self._handles.get(key)
            if existing and existing.is_alive:
                return existing.status.to_dict()
            token = self._token_store.get_user_token(user_id=user_id, org_id=org_id, include_secret=True)
            if not token:
                raise ValueError("当前用户还没有保存飞书 App 凭证")
            if token.get("credential_type") != "app_credentials":
                raise ValueError("飞书消息桥需要 App ID + App Secret 模式")
            app_id = str(token.get("app_id") or "").strip()
            app_secret = str(token.get("app_secret") or "").strip()
            if not app_id or not app_secret:
                raise ValueError("飞书 App ID 或 App Secret 为空")
            handle = _BridgeHandle(
                app_id=app_id,
                app_secret=app_secret,
                user_id=user_id,
                org_id=org_id,
                domain=self._domain,
                runtime_factory=self._runtime_factory,
                api_tool_factory=self._api_tool_factory,
            )
            self._handles[key] = handle
            handle.start()
            return handle.status.to_dict()

    def stop(self, *, user_id: str, org_id: str) -> dict[str, Any]:
        key = _key(user_id, org_id)
        with self._lock:
            handle = self._handles.get(key)
            if not handle:
                return FeishuBridgeStatus(user_id=user_id, org_id=org_id).to_dict()
            handle.stop()
            if not handle.is_alive:
                self._handles.pop(key, None)
            return handle.status.to_dict()

    def status(self, *, user_id: str, org_id: str) -> dict[str, Any]:
        key = _key(user_id, org_id)
        with self._lock:
            handle = self._handles.get(key)
            if not handle:
                return FeishuBridgeStatus(user_id=user_id, org_id=org_id).to_dict()
            return handle.status.to_dict()

    def logs(self, *, user_id: str, org_id: str, limit: int = 100) -> dict[str, Any]:
        key = _key(user_id, org_id)
        with self._lock:
            handle = self._handles.get(key)
            if not handle:
                return {"user_id": user_id, "org_id": org_id, "logs": []}
            return {"user_id": user_id, "org_id": org_id, "logs": handle.get_logs(limit=limit)}


class _BridgeHandle:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        user_id: str,
        org_id: str,
        domain: str,
        runtime_factory: RuntimeFactory,
        api_tool_factory: Callable[[], FeishuApiTool],
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._user_id = user_id
        self._org_id = org_id
        self._domain = domain
        self._runtime_factory = runtime_factory
        self._api_tool_factory = api_tool_factory
        self.status = FeishuBridgeStatus(user_id=user_id, org_id=org_id)
        self._status_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_loop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(target=self._run_worker_loop, name=f"feishu-agent-worker-{user_id}", daemon=True)
        self._ws_thread = threading.Thread(target=self._run_ws_client, name=f"feishu-ws-{user_id}", daemon=True)
        self._sdk_loop: Any = None
        self._sdk_client: Any = None
        self._processed: dict[str, float] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._logs: list[dict[str, Any]] = []

    @property
    def is_alive(self) -> bool:
        return self._worker_thread.is_alive() or self._ws_thread.is_alive()

    def start(self) -> None:
        now = time.time()
        with self._status_lock:
            self.status.state = "starting"
            self.status.running = True
            self.status.started_at = now
            self.status.stopped_at = None
            self.status.last_error = ""
        self._log("bridge_starting")
        self._worker_thread.start()
        self._ws_thread.start()
        asyncio.run_coroutine_threadsafe(self._prefetch_bot_identity(), self._worker_loop)

    def stop(self) -> None:
        self._stop_event.set()
        self._log("bridge_stopping")
        with self._status_lock:
            self.status.state = "stopping"
            self.status.running = False
        if self._sdk_loop is not None:
            with contextlib.suppress(Exception):
                if self._sdk_client is not None and hasattr(self._sdk_client, "_disconnect"):
                    asyncio.run_coroutine_threadsafe(self._sdk_client._disconnect(), self._sdk_loop)
                self._sdk_loop.call_soon_threadsafe(self._sdk_loop.stop)
        self._worker_loop.call_soon_threadsafe(self._worker_loop.stop)
        self._ws_thread.join(timeout=2)
        self._worker_thread.join(timeout=2)
        with self._status_lock:
            self.status.state = "stopped"
            self.status.running = False
            self.status.stopped_at = time.time()
        self._log("bridge_stopped")

    def get_logs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), _MAX_DIAGNOSTIC_LOGS))
        with self._status_lock:
            return list(self._logs[-limit:])

    def _log(self, event: str, level: str = "info", **fields: Any) -> None:
        entry = {
            "ts": time.time(),
            "ts_iso": _iso_from_ts(time.time()),
            "level": level,
            "event": event,
            **{key: _safe_log_value(value) for key, value in fields.items()},
        }
        with self._status_lock:
            self._logs.append(entry)
            if len(self._logs) > _MAX_DIAGNOSTIC_LOGS:
                del self._logs[: len(self._logs) - _MAX_DIAGNOSTIC_LOGS]
        log_method = logger.warning if level in {"warning", "error"} else logger.info
        log_method("feishu_ws_bridge %s %s", event, {k: v for k, v in entry.items() if k not in {"ts", "ts_iso", "event"}})

    def enqueue_event(self, data: Any) -> None:
        if self._stop_event.is_set():
            self._log("event_drop_stopped", level="warning", data_type=type(data).__name__)
            return
        self._log("event_enqueued", data_type=type(data).__name__)
        asyncio.run_coroutine_threadsafe(self._handle_message_event(data), self._worker_loop)

    def _run_worker_loop(self) -> None:
        asyncio.set_event_loop(self._worker_loop)
        self._log("worker_loop_started")
        try:
            self._worker_loop.run_forever()
        finally:
            self._log("worker_loop_stopping")
            pending = asyncio.all_tasks(self._worker_loop)
            for task in pending:
                task.cancel()
            if pending:
                self._worker_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._worker_loop.close()
            self._log("worker_loop_closed")

    def _run_ws_client(self) -> None:
        try:
            self._log("ws_client_importing")
            from lark_oapi import EventDispatcherHandler, LogLevel, ws
            from lark_oapi.ws import client as ws_client_module

            self._sdk_loop = ws_client_module.loop

            def on_message(data: Any) -> None:
                self._log("ws_message_received", data_type=type(data).__name__)
                self.enqueue_event(data)

            self._log("ws_dispatcher_building")
            dispatcher = (
                EventDispatcherHandler.builder("", "", LogLevel.INFO)
                .register_p2_im_message_receive_v1(on_message)
                .build()
            )
            self._log("ws_client_building", domain=self._domain)
            self._sdk_client = ws.Client(
                self._app_id,
                self._app_secret,
                log_level=LogLevel.INFO,
                event_handler=dispatcher,
                domain=self._domain,
                auto_reconnect=True,
                source="my-agent-core-feishu-bridge",
                extra_ua_tags=["agent-bridge"],
            )
            with self._status_lock:
                self.status.state = "running"
                self.status.running = True
            self._log("ws_client_starting")
            self._sdk_client.start()
            self._log("ws_client_exited", level="warning")
        except Exception as exc:
            if not self._stop_event.is_set():
                self._log("ws_client_failed", level="error", error=_redact_error(str(exc)))
                with self._status_lock:
                    self.status.state = "failed"
                    self.status.running = False
                    self.status.last_error = _redact_error(str(exc))
        finally:
            if self._stop_event.is_set():
                with self._status_lock:
                    self.status.state = "stopped"
                    self.status.running = False
                    self.status.stopped_at = time.time()
                self._log("ws_client_stopped")

    async def _prefetch_bot_identity(self) -> None:
        try:
            self._log("bot_identity_prefetch_start")
            result = await self._call_feishu_api("GET", "/open-apis/bot/v3/info")
            payload = _parse_tool_payload(result)
            data = None
            if isinstance(payload, dict):
                data = payload.get("data") if isinstance(payload.get("data"), dict) else payload.get("bot")
            if isinstance(data, dict):
                with self._status_lock:
                    self.status.bot_open_id = str(data.get("open_id") or "")
                    self.status.bot_name = str(data.get("app_name") or data.get("name") or "")
                self._log("bot_identity_prefetch_success", bot_open_id=self.status.bot_open_id, bot_name=self.status.bot_name)
            else:
                self._log("bot_identity_prefetch_empty", level="warning", payload=payload)
        except Exception as exc:
            self._log("bot_identity_prefetch_failed", level="error", error=_redact_error(str(exc)))
            with self._status_lock:
                self.status.last_error = _redact_error(str(exc))

    async def _handle_message_event(self, data: Any) -> None:
        started = time.time()
        try:
            inbound = _extract_inbound_message(data)
            if inbound is None:
                self._log("message_parse_ignored", level="warning", data_type=type(data).__name__)
                self._mark_ignored()
                return
            self._log(
                "message_received",
                message_id=inbound.message_id,
                chat_id=inbound.chat_id,
                chat_type=inbound.chat_type,
                message_type=inbound.message_type,
                sender_type=inbound.sender_type,
                sender_open_id=inbound.sender_open_id,
                text_len=len(inbound.text),
                mentions=len(inbound.mentions),
            )
            with self._status_lock:
                self.status.last_event_at = time.time()
                self.status.last_message_id = inbound.message_id
                self.status.message_count += 1
            ignored_reason = self._ignore_reason(inbound)
            if ignored_reason:
                self._log("message_ignored", reason=ignored_reason, message_id=inbound.message_id, chat_type=inbound.chat_type)
                self._mark_ignored(ignored_reason)
                return
            self._record_processed(inbound.message_id)
            self._log("message_processing_start", message_id=inbound.message_id)
            session_id = ""

            try:
                session_id = _session_id_for_message(inbound, user_id=self._user_id, org_id=self._org_id)
                with self._status_lock:
                    self.status.last_session_id = session_id
                    self.status.last_ignored_reason = ""
                lock = self._session_locks.setdefault(session_id, asyncio.Lock())
                async with lock:
                    self._log("agent_run_start", message_id=inbound.message_id, session_id=session_id)
                    prompt = _build_agent_prompt(inbound)
                    runtime = self._runtime_factory(
                        session_id=session_id,
                        ask_callback=None,
                        user_id=self._user_id,
                        org_id=self._org_id,
                    )
                    try:
                        assistant_text = await _collect_assistant_text(runtime, prompt)
                    except Exception as exc:
                        self._log("agent_run_failed", level="error", message_id=inbound.message_id, session_id=session_id, error=_redact_error(str(exc)))
                        await self._send_reply(inbound, f"处理这条消息时出错了：{_redact_error(str(exc))}")
                        raise
                    self._log("agent_run_success", message_id=inbound.message_id, session_id=session_id, reply_len=len(assistant_text))
                    if not assistant_text:
                        assistant_text = "我这边没有生成可发送的回复。"
                    await self._send_reply(inbound, assistant_text)
            except Exception:
                self._release_processed(inbound.message_id)
                self._log("message_processing_failed", level="error", message_id=inbound.message_id, duration_ms=int((time.time() - started) * 1000))
                raise
            with self._status_lock:
                self.status.last_reply_at = time.time()
                self.status.reply_count += 1
                self.status.last_error = ""
            self._log("message_processing_success", message_id=inbound.message_id, duration_ms=int((time.time() - started) * 1000))
        except Exception as exc:
            self._log("message_handler_failed", level="error", error=_redact_error(str(exc)))
            with self._status_lock:
                self.status.last_error = _redact_error(str(exc))

    def _ignore_reason(self, inbound: FeishuInboundMessage) -> str:
        if inbound.sender_type == "app":
            return "sender_is_app"
        bot_open_id = self.status.bot_open_id.strip()
        if bot_open_id and inbound.sender_open_id == bot_open_id:
            return "sender_is_current_bot"
        if inbound.message_id in self._processed:
            return "duplicate_message"
        if not inbound.text.strip():
            return "empty_message"
        # Direct chats are always addressed to the bot. Groups require an @bot
        # mention once bot identity is known, matching OpenClaw's safe default.
        if inbound.chat_type in {"p2p", "private"}:
            return ""
        if bot_open_id:
            return "" if any(_mention_open_id(item) == bot_open_id for item in inbound.mentions) else "group_without_bot_mention"
        return ""

    def _record_processed(self, message_id: str) -> None:
        now = time.time()
        self._processed[message_id] = now
        cutoff = now - _PROCESSED_TTL_SECONDS
        for key, ts in list(self._processed.items()):
            if ts < cutoff:
                self._processed.pop(key, None)
        if len(self._processed) > _MAX_PROCESSED_MESSAGES:
            for key, _ in sorted(self._processed.items(), key=lambda item: item[1])[: len(self._processed) - _MAX_PROCESSED_MESSAGES]:
                self._processed.pop(key, None)

    def _release_processed(self, message_id: str) -> None:
        self._processed.pop(message_id, None)

    def _mark_ignored(self, reason: str = "") -> None:
        with self._status_lock:
            self.status.ignored_count += 1
            self.status.last_ignored_reason = reason

    async def _send_reply(self, inbound: FeishuInboundMessage, text: str) -> None:
        content = json.dumps({"text": text}, ensure_ascii=False)
        body = {"msg_type": "text", "content": content}
        reply_path = f"/open-apis/im/v1/messages/{urllib.parse.quote(inbound.message_id, safe='')}/reply"
        self._log("reply_primary_start", message_id=inbound.message_id, path=reply_path, text_len=len(text))
        result = await self._call_feishu_api("POST", reply_path, body=body)
        payload = _parse_tool_payload(result)
        if not result.is_error and _is_feishu_success(payload):
            self._log("reply_primary_success", message_id=inbound.message_id, code=payload.get("code"), status=payload.get("status"))
            return
        self._log(
            "reply_primary_failed",
            level="warning",
            message_id=inbound.message_id,
            is_error=bool(result.is_error),
            code=payload.get("code"),
            status=payload.get("status"),
            msg=payload.get("msg") or payload.get("message"),
        )
        fallback_body = {"receive_id": inbound.chat_id, **body}
        self._log("reply_fallback_start", message_id=inbound.message_id, chat_id=inbound.chat_id)
        fallback = await self._call_feishu_api(
            "POST",
            "/open-apis/im/v1/messages",
            query={"receive_id_type": "chat_id"},
            body=fallback_body,
        )
        fallback_payload = _parse_tool_payload(fallback)
        if fallback.is_error or not _is_feishu_success(fallback_payload):
            self._log(
                "reply_fallback_failed",
                level="error",
                message_id=inbound.message_id,
                is_error=bool(fallback.is_error),
                code=fallback_payload.get("code"),
                status=fallback_payload.get("status"),
                msg=fallback_payload.get("msg") or fallback_payload.get("message"),
            )
            raise RuntimeError(f"Feishu reply failed: {fallback.content}")
        self._log("reply_fallback_success", message_id=inbound.message_id, code=fallback_payload.get("code"), status=fallback_payload.get("status"))

    async def _call_feishu_api(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ):
        result = await self._api_tool_factory().call(
            {"method": method, "path": path, **({"query": query} if query else {}), **({"body": body} if body is not None else {})},
            ToolContext(
                session_id="feishu-websocket-bridge",
                messages=[],
                metadata={"user_id": self._user_id, "org_id": self._org_id},
            ),
        )
        payload = _parse_tool_payload(result)
        self._log(
            "feishu_api_call",
            method=method,
            path=path,
            is_error=bool(result.is_error),
            code=payload.get("code"),
            status=payload.get("status"),
            msg=payload.get("msg") or payload.get("message"),
        )
        return result


async def _collect_assistant_text(runtime: AgentRuntime, prompt: str) -> str:
    chunks: list[str] = []
    async for event in runtime.run(prompt):
        if event.type == "assistant_text":
            text = str(event.data.get("text") or "").strip()
            if text:
                chunks.append(text)
        elif event.type == "loop_failed":
            raise RuntimeError(str(event.data.get("error") or "agent loop failed"))
    return "\n".join(chunks).strip()


def _extract_inbound_message(data: Any) -> FeishuInboundMessage | None:
    event = _read(data, "event")
    sender = _read(event, "sender")
    message = _read(event, "message")
    if sender is None or message is None:
        return None
    message_id = _string(_read(message, "message_id"))
    chat_id = _string(_read(message, "chat_id"))
    chat_type = _string(_read(message, "chat_type"))
    message_type = _string(_read(message, "message_type"))
    raw_content = _string(_read(message, "content"))
    if not message_id or not chat_id or not chat_type or not message_type:
        return None
    sender_id = _read(sender, "sender_id")
    mentions = [_object_to_dict(item) for item in (_read(message, "mentions") or [])]
    return FeishuInboundMessage(
        message_id=message_id,
        chat_id=chat_id,
        chat_type=chat_type,
        message_type=message_type,
        text=_parse_message_text(raw_content, message_type, mentions=mentions),
        sender_type=_string(_read(sender, "sender_type")),
        sender_open_id=_string(_read(sender_id, "open_id")),
        sender_user_id=_string(_read(sender_id, "user_id")),
        root_id=_string(_read(message, "root_id")),
        thread_id=_string(_read(message, "thread_id")),
        mentions=mentions,
    )


def _parse_message_text(content: str, message_type: str, *, mentions: list[dict[str, Any]]) -> str:
    if message_type == "post":
        return _parse_post_text(content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    if message_type == "text" and isinstance(parsed, dict):
        text = str(parsed.get("text") or "")
        return _strip_mention_keys(text, mentions).strip()
    if isinstance(parsed, dict) and message_type == "audio":
        speech = str(parsed.get("speech_to_text") or "").strip()
        if speech:
            return speech
    if isinstance(parsed, dict):
        file_name = str(parsed.get("file_name") or "").strip()
        return f"[{message_type}{': ' + file_name if file_name else ''}]"
    return content.strip()


def _parse_post_text(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    lines: list[str] = []
    localized_payloads = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("content"), list):
            localized_payloads.append(parsed)
        localized_payloads.extend(item for item in parsed.values() if isinstance(item, dict))
    for payload in localized_payloads:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, list):
                continue
            parts = []
            for item in block:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("name") or item.get("href") or ""))
            line = "".join(parts).strip()
            if line:
                lines.append(line)
    return "\n".join(lines).strip()


def _build_agent_prompt(inbound: FeishuInboundMessage) -> str:
    context = {
        "source": "feishu",
        "message_id": inbound.message_id,
        "chat_id": inbound.chat_id,
        "chat_type": inbound.chat_type,
        "sender_open_id": inbound.sender_open_id,
        "sender_user_id": inbound.sender_user_id,
    }
    return (
        "你正在通过飞书机器人收到一条用户消息。请直接生成要回复给用户的最终文本；"
        "不要调用 FeishuApi 发送消息，系统会把你的最终回复自动发回同一个飞书会话。\n\n"
        f"上下文：{json.dumps(context, ensure_ascii=False)}\n\n"
        f"用户消息：\n{inbound.text}"
    )


def _strip_mention_keys(text: str, mentions: list[dict[str, Any]]) -> str:
    result = text
    for mention in mentions:
        key = str(mention.get("key") or "")
        if key:
            result = result.replace(key, "")
    return result


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): _object_to_dict(v) if _is_object_like(v) else v for k, v in value.items()}
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        item = getattr(value, name, None)
        if callable(item):
            continue
        if _is_object_like(item):
            result[name] = _object_to_dict(item)
        elif item is not None:
            result[name] = item
    return result


def _is_object_like(value: Any) -> bool:
    return isinstance(value, dict) or hasattr(value, "__dict__")


def _mention_open_id(mention: dict[str, Any]) -> str:
    mention_id = mention.get("id")
    return str(mention_id.get("open_id") if isinstance(mention_id, dict) else "").strip()


def _parse_tool_payload(result: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(result.content or "{}"))
    except json.JSONDecodeError:
        return {"message": str(result.content or "")}
    return payload if isinstance(payload, dict) else {"data": payload}


def _is_feishu_success(payload: dict[str, Any]) -> bool:
    status = int(payload.get("status") or 200)
    return status < 400 and payload.get("code") in (None, 0)


def _read(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _session_id_for_message(inbound: FeishuInboundMessage, *, user_id: str = "", org_id: str = "") -> str:
    # Mirror OpenClaw's routing shape: DMs are scoped by Feishu sender open_id,
    # while groups default to the chat conversation. Do not include message_id;
    # otherwise every inbound message becomes a fresh AgentRuntime session.
    if inbound.chat_type in {"p2p", "private"}:
        conversation_id = inbound.sender_open_id or inbound.chat_id
        conversation_kind = "direct"
    else:
        conversation_id = inbound.chat_id
        conversation_kind = "group"
    raw = f"feishu-{org_id or 'org'}-{user_id or 'user'}-{conversation_kind}-{conversation_id}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-._")[:96]


def _key(user_id: str, org_id: str) -> tuple[str, str]:
    return str(user_id or ""), str(org_id or "")


def _redact_error(text: str) -> str:
    return re.sub(r"(?i)(token|secret|authorization|password)[=: ]+[^\s,;]+", r"\1=[redacted]", text)[:500]


def _safe_log_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_error(value)[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:30]:
            key_text = str(key)
            if re.search(r"(?i)(token|secret|authorization|password)", key_text):
                result[key_text] = "[redacted]"
            else:
                result[key_text] = _safe_log_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(item) for item in list(value)[:30]]
    return _redact_error(str(value))[:500]


def _iso_from_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()
