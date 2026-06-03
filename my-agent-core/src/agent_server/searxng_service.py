"""Bundled SearXNG service lifecycle.

SearXNG is vendored under ``vendor/searxng`` and can be started alongside the
FastAPI app to provide a free self-hosted metasearch backend for WebSearch.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SEARXNG_BOOTSTRAP = """
import concurrent.futures.thread
from searx.webapp import run

run()
""".strip()


@dataclass(slots=True)
class SearxngStatus:
    enabled: bool
    running: bool
    endpoint: str
    source_dir: str
    settings_path: str
    pid: int | None = None
    reason: str = ""


class SearxngService:
    def __init__(self, *, project_root: Path | None = None) -> None:
        self._project_root = project_root or Path(__file__).resolve().parents[2]
        self._source_dir = Path(os.getenv("AGENT_SEARXNG_SOURCE_DIR") or self._project_root / "vendor" / "searxng").expanduser().resolve()
        self._state_dir = Path(os.getenv("AGENT_SEARXNG_STATE_DIR") or Path.home() / ".my-agent-core" / "searxng").expanduser().resolve()
        self._settings_path = Path(os.getenv("AGENT_SEARXNG_SETTINGS_PATH") or self._state_dir / "settings.yml").expanduser().resolve()
        self._host = os.getenv("AGENT_SEARXNG_HOST") or "127.0.0.1"
        self._port = int(os.getenv("AGENT_SEARXNG_PORT") or "18888")
        self._process: subprocess.Popen[str] | None = None
        self._last_reason = ""

    @property
    def endpoint(self) -> str:
        return os.getenv("AGENT_SEARXNG_ENDPOINT") or f"http://{self._host}:{self._port}"

    @property
    def search_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/search"

    def is_enabled(self) -> bool:
        return _env_truthy("AGENT_SEARXNG_ENABLED", default=True)

    def status(self) -> SearxngStatus:
        running = self._process is not None and self._process.poll() is None
        return SearxngStatus(
            enabled=self.is_enabled(),
            running=running,
            endpoint=self.endpoint,
            source_dir=str(self._source_dir),
            settings_path=str(self._settings_path),
            pid=self._process.pid if running and self._process else None,
            reason=self._last_reason,
        )

    async def start(self) -> SearxngStatus:
        if not self.is_enabled():
            self._last_reason = "disabled by AGENT_SEARXNG_ENABLED"
            return self.status()
        if self._process is not None and self._process.poll() is None:
            self._publish_endpoint_env()
            return self.status()
        if not self._source_dir.exists():
            self._last_reason = f"SearXNG source directory not found: {self._source_dir}"
            return self.status()

        missing = self._missing_dependencies()
        if missing:
            self._last_reason = (
                "missing SearXNG dependencies: "
                + ", ".join(missing)
                + f"; install with: python -m pip install -r {self._source_dir / 'requirements.txt'} -r {self._source_dir / 'requirements-server.txt'}"
            )
            return self.status()

        self._write_settings()
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(self._source_dir) + os.pathsep + env.get("PYTHONPATH", ""),
                "SEARXNG_SETTINGS_PATH": str(self._settings_path),
                "SEARXNG_BIND_ADDRESS": self._host,
                "SEARXNG_PORT": str(self._port),
                "SEARXNG_SECRET": env.get("SEARXNG_SECRET") or env.get("AGENT_SEARXNG_SECRET") or secrets.token_urlsafe(32),
                "SEARXNG_DEBUG": env.get("SEARXNG_DEBUG", "false"),
            }
        )
        self._process = subprocess.Popen(
            [sys.executable, "-c", _SEARXNG_BOOTSTRAP],
            cwd=str(self._source_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._publish_endpoint_env()
        ready = await self._wait_until_ready(timeout=15)
        if ready:
            self._last_reason = "started"
        else:
            poll = self._process.poll() if self._process else None
            if poll is not None:
                stderr = ""
                try:
                    stderr = (self._process.stderr.read() if self._process.stderr else "")[-1200:]
                except Exception:
                    pass
                self._last_reason = f"process exited with {poll}: {stderr}".strip()
            else:
                self._last_reason = "started but readiness check timed out"
        return self.status()

    async def stop(self) -> None:
        if self._process is None:
            return
        proc = self._process
        self._process = None
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, 5)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait, 5)

    def _publish_endpoint_env(self) -> None:
        # Let WebSearch auto-discover the bundled instance unless the user has
        # explicitly configured a different provider/endpoint.
        os.environ.setdefault("AGENT_WEB_SEARCH_PROVIDER", "searxng")
        os.environ.setdefault("AGENT_WEB_SEARCH_ENDPOINT", self.endpoint)

    def _missing_dependencies(self) -> list[str]:
        missing: list[str] = []
        for module in ("flask", "flask_babel", "lxml", "httpx", "yaml"):
            try:
                __import__(module)
            except Exception:
                missing.append(module)
        return missing

    def _write_settings(self) -> None:
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(
            "\n".join(
                [
                    "use_default_settings: true",
                    "general:",
                    '  instance_name: "MyAgent SearXNG"',
                    "  enable_metrics: false",
                    "search:",
                    "  formats:",
                    "    - html",
                    "    - json",
                    "server:",
                    f'  bind_address: "{self._host}"',
                    f"  port: {self._port}",
                    "  limiter: false",
                    "  public_instance: false",
                    f'  secret_key: "{secrets.token_urlsafe(32)}"',
                    "  method: \"GET\"",
                    "ui:",
                    "  static_use_hash: true",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    async def _wait_until_ready(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        url = self.search_endpoint + "?" + urllib.parse.urlencode({"q": "searxng", "format": "json"})
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            try:
                await asyncio.to_thread(_http_get_status, url)
                return True
            except Exception:
                await asyncio.sleep(0.5)
        return False


def _http_get_status(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "my-agent-core searxng healthcheck"})
    with urllib.request.urlopen(req, timeout=2) as response:
        return int(response.status)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


searxng_service = SearxngService()
