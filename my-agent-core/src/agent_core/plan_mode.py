from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from agent_core.permissions.policy import get_session_permission_state, normalize_permission_mode
from agent_core.types import Message, TextBlock, ToolResultBlock


_WORDS = (
    "amber", "anchor", "bamboo", "beacon", "cedar", "cobalt", "comet", "copper",
    "delta", "ember", "falcon", "forest", "harbor", "lantern", "maple", "meadow",
    "meteor", "orchid", "pebble", "prairie", "quartz", "raven", "river", "saffron",
    "silver", "summit", "violet", "willow", "zephyr",
)
_MAX_SLUG_RETRIES = 10
PLAN_METADATA_KEY = "plan_mode"


def get_plans_directory(cwd: str | Path | None = None) -> Path:
    """Return the directory used for session plan files, creating it if needed."""
    configured = os.getenv("AGENT_PLANS_DIR") or os.getenv("MY_AGENT_PLANS_DIR")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(cwd or os.getcwd()).expanduser().resolve() / path
    else:
        path = Path.home() / ".my-agent" / "plans"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _generate_slug() -> str:
    return f"{random.choice(_WORDS)}-{random.choice(_WORDS)}-{random.randint(100, 999)}"


def get_plan_slug(session_id: str | None, *, cwd: str | Path | None = None) -> str:
    state = get_session_permission_state(session_id)
    if state.plan_slug:
        return state.plan_slug
    plans_dir = get_plans_directory(cwd)
    slug = _generate_slug()
    for _ in range(_MAX_SLUG_RETRIES):
        slug = _generate_slug()
        if not (plans_dir / f"{slug}.md").exists():
            break
    state.plan_slug = slug
    return slug


def clear_plan_slug(session_id: str | None) -> None:
    state = get_session_permission_state(session_id)
    state.plan_slug = ""


def clear_plan_state(session_id: str | None) -> None:
    """Clear persisted/in-memory plan state for a session."""
    state = get_session_permission_state(session_id)
    state.pre_plan_mode = ""
    state.plan_slug = ""
    state.plan_accepted = False
    state.plan_clear_context = True
    state.plan_accept_mode = ""


def get_plan_file_path(session_id: str | None, *, cwd: str | Path | None = None) -> Path:
    return get_plans_directory(cwd) / f"{get_plan_slug(session_id, cwd=cwd)}.md"


def get_plan(session_id: str | None, *, cwd: str | Path | None = None) -> str | None:
    path = get_plan_file_path(session_id, cwd=cwd)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def write_plan(session_id: str | None, content: str, *, cwd: str | Path | None = None) -> Path:
    path = get_plan_file_path(session_id, cwd=cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def prepare_context_for_plan_mode(session_id: str | None) -> tuple[str, str]:
    """Switch a session into plan mode, preserving the previous mode once."""
    state = get_session_permission_state(session_id)
    if state.mode != "plan":
        state.pre_plan_mode = state.mode
    state.mode = "plan"
    return state.mode, state.pre_plan_mode or "default"


def exit_plan_mode(session_id: str | None, mode: str | None = None) -> str:
    """Leave plan mode and restore the requested or pre-plan permission mode."""
    state = get_session_permission_state(session_id)
    target = normalize_permission_mode(mode or state.pre_plan_mode or "default")
    if target == "plan":
        target = "default"
    state.mode = target
    state.pre_plan_mode = ""
    return target


def consume_plan_acceptance(session_id: str | None) -> dict[str, object] | None:
    """Return and clear the latest ExitPlanMode approval metadata for a session."""
    state = get_session_permission_state(session_id)
    if not state.plan_accepted:
        return None
    payload = {
        "clearContext": state.plan_clear_context,
        "mode": normalize_permission_mode(str(state.plan_accept_mode or state.mode or "default")),
    }
    state.plan_accepted = False
    state.plan_accept_mode = ""
    return payload


def build_plan_implementation_prompt(
    plan: str,
    *,
    plan_file_path: str | Path | None = None,
    clear_context: bool = True,
) -> str:
    """Build the user message that starts implementation after plan approval."""
    detail_hint = ""
    if clear_context:
        detail_hint = (
            "\n\nContext before plan approval was cleared. If you need exact prior details, "
            "re-read the relevant files from the workspace rather than relying on memory."
        )
    file_hint = f"\n\nPlan file: {plan_file_path}" if plan_file_path else ""
    return f"Implement the following approved plan:\n\n{plan}{file_hint}{detail_hint}"


def export_plan_metadata(session_id: str | None, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Serialize plan-related session state into SessionStore metadata."""
    state = get_session_permission_state(session_id)
    payload: dict[str, Any] = {
        "mode": state.mode,
        "pre_plan_mode": state.pre_plan_mode or "",
        "plan_slug": state.plan_slug or "",
    }
    if state.plan_slug:
        plan_path = get_plan_file_path(session_id, cwd=cwd)
        payload.update({
            "plan_file_path": str(plan_path),
            "plan_exists": plan_path.exists(),
        })
    return payload


def merge_plan_metadata(metadata: dict[str, Any] | None, session_id: str | None, *, cwd: str | Path | None = None) -> dict[str, Any]:
    """Return metadata with current plan state under the plan_mode key."""
    merged = dict(metadata or {})
    plan_meta = export_plan_metadata(session_id, cwd=cwd)
    if plan_meta.get("mode") == "default" and not plan_meta.get("pre_plan_mode") and not plan_meta.get("plan_slug"):
        merged.pop(PLAN_METADATA_KEY, None)
    else:
        merged[PLAN_METADATA_KEY] = plan_meta
    return merged


def persist_plan_metadata(store: Any, session_id: str | None, *, cwd: str | Path | None = None) -> None:
    """Best-effort write-through of current plan state into a SessionStore."""
    if not store or not session_id:
        return
    info = store.get_session(session_id)
    metadata = dict((info or {}).get("metadata") or {})
    store.update_session_metadata(session_id, merge_plan_metadata(metadata, session_id, cwd=cwd))


def restore_plan_state(
    session_id: str | None,
    metadata: dict[str, Any] | None,
    messages: list[Message] | None = None,
    *,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Restore plan mode/session slug from SessionStore metadata and recover missing plan files."""
    plan_meta = dict((metadata or {}).get(PLAN_METADATA_KEY) or {})
    # Backward/diagnostic compatibility: allow flat metadata if callers wrote it.
    if not plan_meta:
        plan_meta = {
            "mode": (metadata or {}).get("permission_mode"),
            "pre_plan_mode": (metadata or {}).get("pre_plan_mode"),
            "plan_slug": (metadata or {}).get("plan_slug"),
        }
    state = get_session_permission_state(session_id)
    restored: dict[str, Any] = {"restored": False, "planRecovered": False}
    mode = plan_meta.get("mode")
    if isinstance(mode, str) and mode:
        state.mode = normalize_permission_mode(mode)
        restored["restored"] = True
    pre_plan_mode = plan_meta.get("pre_plan_mode")
    if isinstance(pre_plan_mode, str):
        state.pre_plan_mode = normalize_permission_mode(pre_plan_mode) if pre_plan_mode else ""
        restored["restored"] = True
    slug = plan_meta.get("plan_slug")
    if isinstance(slug, str) and _is_safe_plan_slug(slug):
        state.plan_slug = slug
        restored["restored"] = True
        restored["planFilePath"] = str(get_plan_file_path(session_id, cwd=cwd))
        if not get_plan(session_id, cwd=cwd):
            recovered = recover_plan_from_messages(messages or [])
            if recovered:
                write_plan(session_id, recovered, cwd=cwd)
                restored["planRecovered"] = True
                restored["recoveredChars"] = len(recovered)
    return restored


def recover_plan_from_messages(messages: list[Message]) -> str | None:
    """Recover an approved/active plan from serialized session messages."""
    for message in reversed(messages):
        for block in reversed(message.content):
            if isinstance(block, TextBlock):
                recovered = _extract_plan_from_text(block.text)
                if recovered:
                    return recovered
            if isinstance(block, ToolResultBlock):
                recovered = _extract_plan_from_text(block.content)
                if recovered:
                    return recovered
    return None


def _extract_plan_from_text(text: str) -> str | None:
    markers = [
        "Implement the following approved plan:\n\n",
        "Plan:\n",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx < 0:
            continue
        plan = text[idx + len(marker):]
        for end_marker in ("\n\nPlan file:", "\n\nContext before plan approval", "\n\nPermission mode:"):
            end = plan.find(end_marker)
            if end >= 0:
                plan = plan[:end]
        plan = plan.strip()
        if plan:
            return plan
    return None


def _is_safe_plan_slug(slug: str) -> bool:
    return bool(slug) and "/" not in slug and "\\" not in slug and ".." not in slug and slug.endswith(".md") is False


def plan_mode_system_prompt(session_id: str | None, *, cwd: str | Path | None = None) -> str:
    state = get_session_permission_state(session_id)
    if state.mode != "plan":
        return ""
    plan_path = get_plan_file_path(session_id, cwd=cwd)
    return (
        "# Plan Mode\n"
        "You are currently in Plan Mode. Explore and design, but do not implement yet.\n"
        "Allowed work: read/search files, inspect the project, update TodoWrite, and write only the plan file.\n"
        "Do not modify project files or run mutating shell commands until the user approves the plan.\n"
        f"Write the final plan to: {plan_path}\n"
        "When the plan is ready, call ExitPlanMode to present it for approval."
    )
