#!/usr/bin/env python3
"""
s03: Permission (豆包 / 火山方舟 Ark 版)

在 s02 基础上加一道权限管线, 工具执行前过三道闸门:
    Gate 1: DENY_LIST     — 硬拒绝 (rm -rf /, sudo, ...), 仅 bash
    Gate 2: PERMISSION_RULES — 规则匹配, 命中则触发 Gate 3
    Gate 3: ask_user      — 暂停等用户确认

agent_loop 主循环结构与 s02 一致, 只在 dispatch 之前插了一段:
    if not check_permission(name, args):
        messages.append({"role": "tool", ..., "content": "Permission denied."})
        continue

s02 的 run_bash 自带的 dangerous 字符串黑名单已经移除 —— 由统一的
check_deny_list 接管, 单一真理来源, 避免双重拦截。

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # 填 ARK_API_KEY 和 MODEL_ID
    python s03_permission/code.py
"""

import json
import os
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.environ["ARK_API_KEY"],
    base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
)
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use tools to solve tasks. Act, don't explain. "
    "All destructive operations require user approval; "
    "if a tool returns 'Permission denied.', try a different approach instead of retrying."
)


# ═══════════════════════════════════════════════════════════
#  s02 原样保留: 工具实现
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """沙箱校验: 把路径展平到绝对路径, 不允许逃出 WORKDIR."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    # 注意: s02 里这里有硬编码黑名单, s03 已交给 check_deny_list 统一处理
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines) if lines else "(empty file)"
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR, recursive=True):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  s02 原样保留: 工具定义 (OpenAI 兼容格式)
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites if exists).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g. **/*.py).",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════
#  s02 原样保留: Dispatch map
# ═══════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
}


# ═══════════════════════════════════════════════════════════
#  s03 新增: 三道闸门
# ═══════════════════════════════════════════════════════════

# Gate 1: 硬拒绝列表 — 永远不允许
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: 规则匹配 — 描述"什么时候需要问用户"
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "")
            for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]


def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: 用户审批 — 规则命中后阻塞等输入
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    try:
        choice = input("   Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    return "allow" if choice in ("y", "yes") else "deny"


# 三道闸门串联 — 协议无关, 直接接收 (name, args)
def check_permission(tool_name: str, args: dict) -> bool:
    # 闸门 1: 硬拒绝 (仅 bash)
    if tool_name == "bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False

    # 闸门 2 + 3: 规则匹配 → 用户审批
    reason = check_rules(tool_name, args)
    if reason:
        decision = ask_user(tool_name, args, reason)
        if decision == "deny":
            return False

    return True


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s02 一致, dispatch 前插入 check_permission
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message

        # 把 assistant 回复追加到历史 (含 tool_calls 字段, 否则下一轮 API 会拒)
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # 模型不再调工具 -> 退出
        if response.choices[0].finish_reason != "tool_calls" or not msg.tool_calls:
            return

        # 按 tool_calls 顺序逐个执行, 结果作为 role=tool 消息追加
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            print(f"\033[33m> {name} {args}\033[0m")

            # s03 新增: 工具执行前过权限管线
            if not check_permission(name, args):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Permission denied.",
                })
                continue

            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                output = f"Error: Unknown tool '{name}'"
            else:
                try:
                    output = handler(**args)
                except TypeError as e:
                    output = f"Error: bad arguments for {name}: {e}"
            print(str(output)[:200])

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output),
            })


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("s03: Permission (豆包版) — 三道闸门管线")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        if last["role"] == "assistant" and last.get("content"):
            print(last["content"])
        print()
