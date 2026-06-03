from __future__ import annotations

from agent_server.codex_credentials import codex_credential_source, delete_codex_credentials, resolve_codex_provider_credentials, save_codex_credentials
from agent_server.model_providers import (
    available_provider_names,
    default_model_for_provider,
    detect_provider_from_env,
    env_model_selection,
    resolve_provider_config,
)


def _clear_codex_env(monkeypatch):
    for name in (
        "AGENT_MODEL_PROVIDER",
        "CODEX_API_KEY",
        "CODEX_PROXY_API_KEY",
        "AGENT_CODEX_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "CODEX_AUTH_JSON",
        "CODEX_BASE_URL",
        "CODEX_MODEL",
        "MODEL_ID",
        "CODEX_HOME",
        "AGENT_CODEX_CREDENTIALS_PATH",
        "AGENT_CODEX_ALLOW_CHATGPT_TOKEN_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_codex_provider_spec_defaults(monkeypatch):
    _clear_codex_env(monkeypatch)

    config = resolve_provider_config("codex")

    assert "codex" in available_provider_names()
    assert config.provider == "codex"
    assert config.model == "gpt-5.4"
    assert config.base_url == "http://localhost:8080/v1"


def test_codex_model_precedence(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.4-high-fast")
    monkeypatch.setenv("MODEL_ID", "fallback-model")

    assert default_model_for_provider("codex") == "gpt-5.4-high-fast"
    assert default_model_for_provider("codex", model_id="explicit-model") == "explicit-model"


def test_codex_credentials_do_not_auto_select_chat_provider(monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("CODEX_API_KEY", "codex-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert detect_provider_from_env() != "codex"
    assert env_model_selection()["model_provider"] != "codex"


def test_codex_base_url_alone_does_not_auto_select_provider(monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("CODEX_BASE_URL", "http://localhost:8080/v1")

    assert detect_provider_from_env() != "codex"


def test_explicit_provider_overrides_codex_detection(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("CODEX_API_KEY", "codex-key")

    assert detect_provider_from_env() == "openai"


def test_explicit_codex_provider_still_supported(monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_API_KEY", "codex-key")

    assert detect_provider_from_env() == "codex"


def test_codex_credentials_from_auth_json_file(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"chatgpt","tokens":{"access_token":"chatgpt-token","refresh_token":"refresh"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert detect_provider_from_env() != "codex"
    assert resolve_codex_provider_credentials() is None
    assert codex_credential_source()["configured"] is True
    assert codex_credential_source()["provider_ready"] is False


def test_codex_env_key_precedes_auth_json_file(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"chatgpt","tokens":{"access_token":"file-token"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CODEX_API_KEY", "proxy-key")

    config = resolve_provider_config("codex")

    assert detect_provider_from_env() != "codex"
    assert config.api_key == "proxy-key"


def test_codex_auth_json_can_be_used_for_provider_with_explicit_escape_hatch(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"chatgpt","tokens":{"access_token":"chatgpt-token"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("AGENT_CODEX_ALLOW_CHATGPT_TOKEN_PROVIDER", "true")

    config = resolve_provider_config("codex")

    assert detect_provider_from_env() != "codex"
    assert config.api_key == "chatgpt-token"


def test_codex_app_managed_credentials_drive_provider_config(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("AGENT_CODEX_CREDENTIALS_PATH", str(tmp_path / "codex-creds.json"))

    save_codex_credentials(
        secret="saved-key",
        credential_type="api_key",
        base_url="http://codex.local/v1",
        model="gpt-5.4-saved",
        image_model="gpt-image-saved",
    )
    config = resolve_provider_config("codex")
    status = codex_credential_source()

    assert detect_provider_from_env() != "codex"
    assert config.api_key == "saved-key"
    assert config.base_url == "http://codex.local/v1"
    assert config.model == "gpt-5.4-saved"
    assert status["configured"] is True
    assert status["source"] == "app:api_key"
    assert status["app_file_exists"] is True

    assert delete_codex_credentials() is True


def test_terminal_codex_credentials_endpoints(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("AGENT_CODEX_CREDENTIALS_PATH", str(tmp_path / "codex-creds.json"))
    import agent_server.app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    headers = {"x-terminal-token": app_module.get_terminal_token()}

    saved = client.put(
        "/terminal/api/integrations/codex",
        headers=headers,
        json={
            "secret": "ui-key",
            "credential_type": "api_key",
            "base_url": "http://ui.local/v1",
            "model": "gpt-ui",
            "image_model": "gpt-image-ui",
        },
    )

    assert saved.status_code == 200, saved.text
    assert saved.json()["codex"]["configured"] is True
    status = client.get("/terminal/api/integrations/codex", headers=headers)
    assert status.status_code == 200
    assert status.json()["source"] == "app:api_key"
    assert status.json()["base_url"] == "http://ui.local/v1"
    assert "ui-key" not in status.text

    deleted = client.delete("/terminal/api/integrations/codex", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_terminal_codex_login_unavailable_returns_clear_error(tmp_path, monkeypatch):
    _clear_codex_env(monkeypatch)
    monkeypatch.setenv("AGENT_CODEX_CREDENTIALS_PATH", str(tmp_path / "codex-creds.json"))
    monkeypatch.setenv("CODEX_CLI_BIN", str(tmp_path / "missing-codex"))
    import agent_server.app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    response = client.post(
        "/terminal/api/integrations/codex/login",
        headers={"x-terminal-token": app_module.get_terminal_token()},
        json={"flow": "browser"},
    )

    assert response.status_code == 400
    assert "Codex CLI not found" in response.text
