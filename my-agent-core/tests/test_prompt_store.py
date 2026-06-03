from __future__ import annotations

import pytest

from agent_core.session.sqlite_store import SQLiteSessionStore
from agent_server.prompt_store import PromptStore, SOUL_DESIGN_PRINCIPLES


def test_prompt_store_renders_design_principles(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)
    store.create_template(
        {
            "metadata": {"name": "soul-test", "description": "test", "version": "1.0.0"},
            "product_name": "MyAgent",
            "base_instructions": "Base instructions.",
            "design_principles": [SOUL_DESIGN_PRINCIPLES[0], {"title": "Disabled", "content": "hidden", "enabled": False}],
            "sections": [{"title": "Rules", "content": "- Do work"}],
        }
    )

    builder = store.create_context_builder("soul-test")

    rendered = "\n\n".join(builder.static_sections or [])
    assert "## Design Principles" in rendered
    assert "Be genuinely helpful" in rendered
    assert "performative filler" in rendered
    assert "Disabled" not in rendered
    assert "## Rules" in rendered


def test_prompt_store_rejects_invalid_design_principles(tmp_path):
    store = PromptStore(prompts_dir=tmp_path)

    with pytest.raises(ValueError, match="design_principles must be a list"):
        store.create_template(
            {
                "metadata": {"name": "bad", "description": "bad", "version": "1.0.0"},
                "base_instructions": "Base",
                "design_principles": {"title": "bad"},
            }
        )


@pytest.mark.asyncio
async def test_runtime_uses_session_prompt_template(tmp_path, monkeypatch):
    import agent_server.runtime_factory as runtime_factory

    prompt_store = PromptStore(prompts_dir=tmp_path / "prompts")
    prompt_store.create_template(
        {
            "metadata": {"name": "default", "description": "default", "version": "1.0.0"},
            "product_name": "MyAgent",
            "base_instructions": "Default base instructions.",
        }
    )
    prompt_store.create_template(
        {
            "metadata": {"name": "terminal-soul", "description": "terminal", "version": "1.0.0"},
            "product_name": "MyAgent",
            "base_instructions": "Terminal custom prompt template.",
        }
    )
    session_store = SQLiteSessionStore(db_path=tmp_path / "sessions.db")
    session_store.create_session(
        session_id="s1",
        user_id="local-user",
        org_id="local-org",
        metadata={"user_id": "local-user", "org_id": "local-org", "prompt_template": "terminal-soul"},
    )

    monkeypatch.setenv("AGENT_MODEL_PROVIDER", "fake")
    monkeypatch.setattr(runtime_factory, "_prompt_store", prompt_store)
    monkeypatch.setattr(runtime_factory, "_session_store", session_store)

    runtime = runtime_factory.create_runtime(session_id="s1", user_id="local-user", org_id="local-org", cwd=tmp_path)
    rendered = "\n".join(await runtime._context_builder.build())

    assert "Terminal custom prompt template." in rendered
    assert "Default base instructions." not in rendered
    assert runtime._config.metadata["prompt_template"] == "terminal-soul"
