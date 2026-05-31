from __future__ import annotations

import argparse
import asyncio
import sys

from agent_server.runtime_factory import create_runtime


def _print(text: str = "") -> None:
    print(text, flush=True)


async def handle_message(message: str, *, mode: str, session_id: str) -> None:
    runtime = create_runtime(mode=mode, session_id=session_id)
    async for event in runtime.run(message):
        if event.type == "text_delta":
            print(event.data.get("text", ""), end="", flush=True)
        elif event.type == "thinking_delta":
            text = event.data.get("text", "")
            if text:
                print(f"\033[90m{text}\033[0m", end="", flush=True)
        elif event.type == "assistant_text":
            _print(f"\033[36m{event.data.get('text', '')}\033[0m")
        elif event.type == "tool_started":
            tool = event.data.get("tool", "?")
            concurrent = " (concurrent)" if event.data.get("concurrent") else ""
            _print(f"\033[33m▸ {tool}{concurrent}\033[0m")
        elif event.type == "tool_completed":
            tool = event.data.get("tool", "?")
            is_error = event.data.get("is_error")
            color = "\033[31m" if is_error else "\033[32m"
            _print(f"{color}✓ {tool}\033[0m")
        elif event.type == "tool_denied":
            _print(f"\033[31m✗ {event.data.get('tool')}: {event.data.get('reason')}\033[0m")
        elif event.type == "permission_request":
            _print(f"\033[33m? Permission required for {event.data.get('tool')}\033[0m")
        elif event.type == "loop_completed":
            _print("\033[32mdone\033[0m")
        elif event.type == "loop_failed":
            _print(f"\033[31merror: {event.data.get('error')}\033[0m")
        elif event.type == "loop_aborted":
            _print("\033[33m[interrupted]\033[0m")


async def main() -> None:
    parser = argparse.ArgumentParser(description="MyAgent terminal REPL")
    parser.add_argument("--mode", default="fake_tool", help="fake | fake_tool | anthropic")
    parser.add_argument("--session-id", default="pty-repl")
    args = parser.parse_args()

    _print("\033[33mMyAgent PTY REPL\033[0m")
    _print("这是一个真实 PTY 里的 Python CLI。输入一句话回车；输入 /exit 退出；Ctrl+C 中断当前输入。")
    _print(f"mode={args.mode} session_id={args.session_id}")

    while True:
        try:
            message = await asyncio.to_thread(input, "\033[32magent> \033[0m")
        except EOFError:
            _print()
            return
        except KeyboardInterrupt:
            _print("^C")
            continue

        message = message.strip()
        if not message:
            continue
        if message in {"/exit", "exit", "quit", ":q"}:
            _print("bye")
            return
        await handle_message(message, mode=args.mode, session_id=args.session_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
