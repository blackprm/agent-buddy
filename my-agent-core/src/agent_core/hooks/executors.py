"""Hooks 执行器 — command (shell) 和 http (webhook) 两种类型。"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from agent_core.hooks.types import (
    HookDefinition,
    HookInput,
    HookOutcome,
    HookResult,
    HookType,
    PermissionBehavior,
)


# ── Command Hook 执行器 ───────────────────────────────────────


async def exec_command_hook(
    hook: HookDefinition,
    hook_input: HookInput,
) -> HookResult:
    """执行 shell 命令钩子。

    流程：
    1. 构建环境变量
    2. 通过 stdin 传入 JSON 输入
    3. 解析 stdout JSON 输出
    4. 退出码语义: 0=成功, 2=阻塞, 其他=非阻塞错误
    """
    import time

    start = time.monotonic()

    # 构建环境变量
    env = os.environ.copy()
    env.update(hook_input.to_env())
    env["CLAUDE_PROJECT_DIR"] = hook_input.cwd  # 兼容 Claude Code 变量名

    # stdin JSON
    stdin_data = json.dumps(hook_input.to_json(), ensure_ascii=False)

    result = HookResult(hook=hook)

    try:
        proc = await asyncio.create_subprocess_exec(
            hook.shell, "-c", hook.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=hook_input.cwd or None,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_data.encode("utf-8")),
                timeout=hook.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            result.outcome = HookOutcome.NonBlockingError
            result.stop_reason = f"Hook timed out after {hook.timeout}s"
            result.duration = time.monotonic() - start
            return result

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        result.stdout = stdout
        result.stderr = stderr
        result.duration = time.monotonic() - start

        # 退出码语义
        if proc.returncode == 2:
            result.outcome = HookOutcome.Blocking
        elif proc.returncode != 0:
            result.outcome = HookOutcome.NonBlockingError
        else:
            result.outcome = HookOutcome.Success

        # 解析 stdout JSON
        _parse_hook_output(stdout.strip(), result)

    except FileNotFoundError as e:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = f"Shell not found: {e}"
        result.duration = time.monotonic() - start
    except Exception as e:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = str(e)
        result.duration = time.monotonic() - start

    return result


# ── HTTP Hook 执行器 ──────────────────────────────────────────


async def exec_http_hook(
    hook: HookDefinition,
    hook_input: HookInput,
) -> HookResult:
    """执行 HTTP POST 钩子。

    流程：
    1. 环境变量插值（仅 allowed_env_vars 中的变量）
    2. POST JSON 输入到配置的 URL
    3. 解析响应 JSON
    """
    import time

    start = time.monotonic()
    result = HookResult(hook=hook)

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = "urllib not available"
        result.duration = time.monotonic() - start
        return result

    # 环境变量插值
    headers = dict(hook.headers)
    for key in list(headers.keys()):
        headers[key] = _interpolate_env(headers[key], hook.allowed_env_vars)

    # 构建 request
    payload = json.dumps(hook_input.to_json(), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        hook.url,
        data=payload,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        loop = asyncio.get_event_loop()
        response_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=hook.timeout)),
            timeout=hook.timeout + 5,
        )
        body = response_bytes.read().decode("utf-8", errors="replace")
        result.stdout = body
        result.outcome = HookOutcome.Success
        _parse_hook_output(body.strip(), result)

    except urllib.error.HTTPError as e:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = f"HTTP {e.code}: {e.reason}"
    except asyncio.TimeoutError:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = f"HTTP hook timed out after {hook.timeout}s"
    except Exception as e:
        result.outcome = HookOutcome.NonBlockingError
        result.stop_reason = str(e)

    result.duration = time.monotonic() - start
    return result


# ── 辅助函数 ──────────────────────────────────────────────────


def _parse_hook_output(output: str, result: HookResult) -> None:
    """解析钩子的 stdout JSON 输出。"""
    if not output:
        return

    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        # 非 JSON 输出，作为系统消息
        if result.outcome == HookOutcome.Blocking:
            result.stop_reason = output[:500]
        return

    if not isinstance(parsed, dict):
        return

    # 通用字段
    if parsed.get("continue") is False:
        result.outcome = HookOutcome.Blocking
        result.prevent_continuation = True
        result.stop_reason = parsed.get("stopReason", "Hook blocked continuation")

    if parsed.get("suppressOutput"):
        result.stdout = ""

    if parsed.get("systemMessage"):
        result.system_message = parsed["systemMessage"]

    # 事件专属输出
    specific = parsed.get("hookSpecificOutput")
    if isinstance(specific, dict):
        # 权限决策
        perm = specific.get("permissionDecision")
        if perm in ("allow", "deny", "ask"):
            result.permission_behavior = PermissionBehavior(perm)
            result.permission_reason = specific.get("permissionDecisionReason", "")

        # 修改工具输入
        updated = specific.get("updatedInput")
        if isinstance(updated, dict):
            result.updated_input = updated

        # 额外上下文
        ctx = specific.get("additionalContext")
        if isinstance(ctx, str):
            result.additional_context = ctx

    # 顶层 decision 字段（简化格式）
    decision = parsed.get("decision")
    if decision in ("approve", "allow"):
        result.permission_behavior = PermissionBehavior.Allow
    elif decision in ("block", "deny"):
        result.permission_behavior = PermissionBehavior.Deny
        result.outcome = HookOutcome.Blocking
        result.stop_reason = parsed.get("reason", "Hook denied")

    reason = parsed.get("reason")
    if isinstance(reason, str) and reason:
        result.permission_reason = reason


def _interpolate_env(template: str, allowed_vars: list[str]) -> str:
    """对模板中的 $VAR 或 ${VAR} 进行环境变量插值（仅允许白名单变量）。"""
    if not allowed_vars:
        return template

    def replacer(m: re.Match) -> str:
        var_name = m.group(1) or m.group(2)
        if var_name in allowed_vars:
            return os.environ.get(var_name, "")
        return m.group(0)

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", replacer, template)
