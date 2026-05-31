#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop (豆包 / 火山方舟 Ark 版)

核心模式（与官方一致，仅协议换成 OpenAI 兼容）:

    while finish_reason == "tool_calls":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool result |
                          +---------------+
                          (loop continues)

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # 填 ARK_API_KEY 和 MODEL_ID
    python s01_agent_loop/code.py
"""

import json
import os
import subprocess

try:
    import readline
    # macOS libedit 中文输入退格修复
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 客户端: 指向豆包 / 火山方舟 ─────────────────────────────
# 默认走火山方舟北京区。如有自定义网关，可在 .env 设 ARK_BASE_URL 覆盖。
client = OpenAI(
    api_key=os.environ["ARK_API_KEY"],
    base_url=os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
)
MODEL = os.environ["MODEL_ID"]   # e.g. doubao-seed-1-6-250615 或方舟接入点 ID

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── 工具定义: OpenAI 兼容格式 ──────────────────────────────
TOOLS = [{
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
}]


# ── 工具执行 ───────────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ── 核心循环: 调工具直到模型不再调 ─────────────────────────
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message

        # 把 assistant 回复追加进历史 (OpenAI 协议要求保留 tool_calls 字段)
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

        # 模型不再调工具 -> 结束
        finish_reason = response.choices[0].finish_reason
        if finish_reason != "tool_calls" or not msg.tool_calls:
            return

        # 执行每一个工具调用, 结果作为 role=tool 的消息追加
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            command = args.get("command", "")
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


# ── 入口 ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("s01: Agent Loop (豆包版)")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
