from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CodexCredentials:
    """Resolved credentials for Codex-compatible providers.

    ``api_key`` is secret. Config/status APIs must only expose source metadata.
    """

    api_key: str
    source: str
    auth_mode: str = "unknown"
    auth_file: str = ""
    base_url: str = ""
    model: str = ""
    image_model: str = ""


_LOGIN_LOCK = threading.Lock()
_LOGIN_PROCESS: subprocess.Popen[str] | None = None
_LOGIN_OUTPUT: deque[str] = deque(maxlen=120)
_LOGIN_STARTED_AT: float | None = None
_LOGIN_FLOW = ""


def resolve_codex_credentials() -> CodexCredentials | None:
    """Resolve Codex credentials from app, env, or official Codex auth.json.

    Precedence:
    1. Explicit env API keys for codex-proxy/OpenAI-compatible gateways.
    2. App-managed credentials saved from the terminal/admin UI.
    3. ``CODEX_ACCESS_TOKEN`` for non-interactive access-token flows.
    4. ``CODEX_AUTH_JSON`` for secret-manager based CI/CD flows.
    5. ``${CODEX_HOME:-~/.codex}/auth.json`` written by Codex CLI/Desktop.

    This module never refreshes or writes official Codex auth.json; Codex CLI
    remains responsible for refresh. App-managed credentials are stored in a
    separate MyAgent file.
    """

    app_credentials = _load_app_credentials()
    explicit = _first_env(("CODEX_API_KEY", "CODEX_PROXY_API_KEY", "AGENT_CODEX_API_KEY"))
    if explicit:
        return _with_app_settings(CodexCredentials(api_key=explicit, source="env:CODEX_API_KEY", auth_mode="api_key"), app_credentials)

    if app_credentials:
        return app_credentials

    access_token = _env_value("CODEX_ACCESS_TOKEN")
    if access_token:
        return CodexCredentials(api_key=access_token, source="env:CODEX_ACCESS_TOKEN", auth_mode="chatgpt_access_token")

    inline_auth_json = _env_value("CODEX_AUTH_JSON")
    if inline_auth_json:
        resolved = _credentials_from_auth_payload(inline_auth_json, source="env:CODEX_AUTH_JSON", auth_file="")
        if resolved:
            return resolved

    auth_file = codex_auth_file()
    if auth_file.is_file():
        try:
            resolved = _credentials_from_auth_payload(auth_file.read_text(encoding="utf-8"), source="file:auth.json", auth_file=str(auth_file))
            if resolved:
                return resolved
        except OSError:
            return None
    return None


def resolve_codex_provider_credentials() -> CodexCredentials | None:
    """Resolve credentials safe to use with the OpenAI-compatible chat client.

    Official ``codex login`` produces a ChatGPT/Codex access token for the Codex
    CLI's own backend flow.  It is useful for detecting that the user has logged
    in, but it is not a drop-in OpenAI-compatible API key for
    ``/v1/chat/completions``.  Auto-selecting the codex provider from that token
    makes ordinary chat fail with connection/auth retries.  Therefore provider
    auto-detection only uses gateway-style API keys unless explicitly overridden.
    """

    explicit = _first_env(("CODEX_API_KEY", "CODEX_PROXY_API_KEY", "AGENT_CODEX_API_KEY"))
    app_credentials = _load_app_credentials()
    if explicit:
        return _with_app_settings(CodexCredentials(api_key=explicit, source="env:CODEX_API_KEY", auth_mode="api_key"), app_credentials)
    if app_credentials and app_credentials.source == "app:api_key":
        return app_credentials
    if _env_value("AGENT_CODEX_ALLOW_CHATGPT_TOKEN_PROVIDER") in {"1", "true", "yes", "on"}:
        return resolve_codex_credentials()
    return None


def codex_auth_file() -> Path:
    codex_home = Path(os.getenv("CODEX_HOME") or Path.home() / ".codex").expanduser()
    return codex_home / "auth.json"


def codex_app_credentials_file() -> Path:
    return Path(os.getenv("AGENT_CODEX_CREDENTIALS_PATH") or Path.home() / ".my-agent-core" / "codex" / "credentials.json").expanduser()


def save_codex_credentials(
    *,
    secret: str,
    credential_type: str = "api_key",
    base_url: str = "",
    model: str = "",
    image_model: str = "",
    metadata: dict[str, Any] | None = None,
) -> CodexCredentials:
    """Persist app-managed Codex credentials for UI onboarding."""

    secret = str(secret or "").strip()
    if not secret:
        raise ValueError("secret is required")
    credential_type = str(credential_type or "api_key").strip().lower()
    if credential_type not in {"api_key", "access_token", "auth_json"}:
        raise ValueError("credential_type must be api_key, access_token, or auth_json")
    payload: dict[str, Any] = {
        "credential_type": credential_type,
        "secret": secret,
        "base_url": str(base_url or "").strip(),
        "model": str(model or "").strip(),
        "image_model": str(image_model or "").strip(),
        "metadata": metadata or {},
    }
    path = codex_app_credentials_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
        path.parent.chmod(0o700)
    except OSError:
        pass
    credentials = _credentials_from_app_payload(payload, auth_file=str(path))
    if credentials is None:
        raise RuntimeError("failed to save Codex credentials")
    return credentials


def delete_codex_credentials() -> bool:
    path = codex_app_credentials_file()
    if not path.exists():
        return False
    path.unlink()
    return True


def start_codex_login_flow(*, flow: str = "browser") -> dict[str, Any]:
    """Start the official Codex CLI login flow and poll auth.json afterwards.

    We deliberately delegate OAuth/PKCE and token refresh to the official Codex
    CLI instead of cloning private OAuth details.  Browser flow uses Codex's
    localhost:1455 callback; device flow uses ``codex login --device-auth``.
    """

    global _LOGIN_FLOW, _LOGIN_PROCESS, _LOGIN_STARTED_AT
    flow = str(flow or "browser").strip().lower()
    if flow not in {"browser", "device"}:
        raise ValueError("flow must be browser or device")
    codex_bin = shutil.which(os.getenv("CODEX_CLI_BIN") or "codex")
    if not codex_bin:
        raise FileNotFoundError("Codex CLI not found. Install it first, or paste a codex-proxy API key.")
    with _LOGIN_LOCK:
        if _LOGIN_PROCESS is not None and _LOGIN_PROCESS.poll() is None:
            return codex_login_status()
        _LOGIN_OUTPUT.clear()
        cmd = [codex_bin, "login"]
        if flow == "device":
            cmd.append("--device-auth")
        env = os.environ.copy()
        env.setdefault("CODEX_HOME", str(codex_auth_file().parent))
        _LOGIN_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        _LOGIN_STARTED_AT = time.time()
        _LOGIN_FLOW = flow
        threading.Thread(target=_drain_login_output, args=(_LOGIN_PROCESS,), daemon=True).start()
    return codex_login_status()


def codex_login_status() -> dict[str, Any]:
    with _LOGIN_LOCK:
        process = _LOGIN_PROCESS
        return_code = process.poll() if process is not None else None
        running = process is not None and return_code is None
        output = list(_LOGIN_OUTPUT)
        started_at = _LOGIN_STARTED_AT
        flow = _LOGIN_FLOW
    credentials = codex_credential_source()
    return {
        "running": running,
        "return_code": return_code,
        "flow": flow,
        "started_at": started_at,
        "configured": bool(credentials.get("configured")),
        "codex": credentials,
        "output": output[-40:],
        "callback_url": "http://localhost:1455/auth/callback" if flow == "browser" else "",
    }


def _drain_login_output(process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    try:
        for line in process.stdout:
            text = line.rstrip()
            if text:
                with _LOGIN_LOCK:
                    if process is _LOGIN_PROCESS:
                        _LOGIN_OUTPUT.append(_redact_secretish(text))
    finally:
        with _LOGIN_LOCK:
            if process is _LOGIN_PROCESS:
                _LOGIN_OUTPUT.append(f"codex login exited with code {process.poll()}")


def _redact_secretish(text: str) -> str:
    # Keep URLs/device instructions visible, but avoid accidentally surfacing long tokens.
    parts = text.split()
    redacted = []
    for part in parts:
        if len(part) > 48 and not part.startswith(("http://", "https://")):
            redacted.append(part[:8] + "…" + part[-4:])
        else:
            redacted.append(part)
    return " ".join(redacted)


def codex_base_url(default: str = "http://localhost:8080/v1") -> str:
    settings = _load_app_credentials()
    return _env_value("CODEX_BASE_URL") or (settings.base_url if settings else "") or default


def codex_model(default: str = "gpt-5.4") -> str:
    settings = _load_app_credentials()
    return _env_value("CODEX_MODEL") or _env_value("MODEL_ID") or (settings.model if settings else "") or default


def codex_image_model(default: str = "gpt-image-2") -> str:
    settings = _load_app_credentials()
    return _env_value("CODEX_IMAGE_MODEL") or _env_value("OPENAI_IMAGE_MODEL") or (settings.image_model if settings else "") or default


def codex_credential_source() -> dict[str, str | bool]:
    """Return non-secret credential status for diagnostics/config APIs."""

    credentials = resolve_codex_credentials()
    auth_file = codex_auth_file()
    app_file = codex_app_credentials_file()
    image_configured = bool(
        _env_value("AGENT_IMAGE_MODEL")
        or _env_value("AGENT_IMAGE_BASE_URL")
        or _env_value("AGENT_IMAGE_API_KEY")
        or (_env_value("TOP_AIDP_BASE_URL") and (_env_value("TOP_AIDP_APP_KEY") or _env_value("TOP_APP_KEY")) and (_env_value("TOP_AIDP_APP_SECRET") or _env_value("TOP_APP_SECRET")))
        or resolve_codex_provider_credentials()
        or (_env_value("OPENAI_IMAGE_MODEL") and _env_value("OPENAI_API_KEY"))
    )
    return {
        "configured": credentials is not None,
        "provider_ready": resolve_codex_provider_credentials() is not None,
        "image_configured": image_configured,
        "source": credentials.source if credentials else "",
        "auth_mode": credentials.auth_mode if credentials else "",
        "auth_file": credentials.auth_file if credentials else str(auth_file),
        "auth_file_exists": auth_file.is_file(),
        "app_file": str(app_file),
        "app_file_exists": app_file.is_file(),
        "base_url": codex_base_url(),
        "model": codex_model(),
        "image_model": codex_image_model(),
    }


def _credentials_from_auth_payload(payload: str, *, source: str, auth_file: str) -> CodexCredentials | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        token = payload.strip()
        if token:
            return CodexCredentials(api_key=token, source=source, auth_mode="raw_token", auth_file=auth_file)
        return None
    if not isinstance(data, dict):
        return None

    auth_mode = str(data.get("auth_mode") or data.get("mode") or "unknown")
    token = _extract_codex_secret(data)
    if not token:
        return None
    return CodexCredentials(api_key=token, source=source, auth_mode=auth_mode, auth_file=auth_file)


def _load_app_credentials() -> CodexCredentials | None:
    path = codex_app_credentials_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _credentials_from_app_payload(data, auth_file=str(path))


def _credentials_from_app_payload(data: dict[str, Any], *, auth_file: str) -> CodexCredentials | None:
    credential_type = str(data.get("credential_type") or "api_key").strip().lower()
    secret = str(data.get("secret") or "").strip()
    if not secret:
        return None
    base_url = str(data.get("base_url") or "").strip()
    model = str(data.get("model") or "").strip()
    image_model = str(data.get("image_model") or "").strip()
    if credential_type == "auth_json":
        parsed = _credentials_from_auth_payload(secret, source="app:auth_json", auth_file=auth_file)
        if parsed is None:
            return None
        return CodexCredentials(
            api_key=parsed.api_key,
            source="app:auth_json",
            auth_mode=parsed.auth_mode,
            auth_file=auth_file,
            base_url=base_url,
            model=model,
            image_model=image_model,
        )
    auth_mode = "chatgpt_access_token" if credential_type == "access_token" else "api_key"
    return CodexCredentials(
        api_key=secret,
        source=f"app:{credential_type}",
        auth_mode=auth_mode,
        auth_file=auth_file,
        base_url=base_url,
        model=model,
        image_model=image_model,
    )


def _with_app_settings(credentials: CodexCredentials, app_credentials: CodexCredentials | None) -> CodexCredentials:
    if app_credentials is None:
        return credentials
    return CodexCredentials(
        api_key=credentials.api_key,
        source=credentials.source,
        auth_mode=credentials.auth_mode,
        auth_file=credentials.auth_file,
        base_url=app_credentials.base_url,
        model=app_credentials.model,
        image_model=app_credentials.image_model,
    )


def _extract_codex_secret(data: dict[str, Any]) -> str:
    # API-key login variants used by CLI/gateways and hand-written auth payloads.
    for key in ("api_key", "openai_api_key", "OPENAI_API_KEY", "CODEX_API_KEY"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # ChatGPT/Codex OAuth auth.json shape documented by OpenAI.
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        for key in ("access_token", "id_token"):
            value = tokens.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("access_token", "id_token", "token"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _env_value(name)
        if value:
            return value
    return None


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None
