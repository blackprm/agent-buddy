from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_core.hooks.engine import HookEngine
from agent_core.hooks.types import HookEvent


_VALID_WORKTREE_SLUG_SEGMENT = re.compile(r"^[a-zA-Z0-9._-]+$")
_MAX_WORKTREE_SLUG_LENGTH = 64


def validate_worktree_slug(slug: str) -> None:
    if len(slug) > _MAX_WORKTREE_SLUG_LENGTH:
        raise ValueError(f"Invalid workspace name: must be {_MAX_WORKTREE_SLUG_LENGTH} characters or fewer")
    for segment in slug.split("/"):
        if segment in {".", ".."}:
            raise ValueError(f"Invalid workspace name {slug!r}: must not contain '.' or '..' path segments")
        if not _VALID_WORKTREE_SLUG_SEGMENT.match(segment):
            raise ValueError(
                f"Invalid workspace name {slug!r}: segments must contain only letters, digits, dots, underscores, and dashes"
            )


def flatten_worktree_slug(slug: str) -> str:
    return slug.replace("/", "+")


def worktree_branch_name(slug: str) -> str:
    return f"worktree-{flatten_worktree_slug(slug)}"


@dataclass(slots=True)
class WorktreeInfo:
    session_id: str
    workspace_id: str
    original_cwd: str
    worktree_path: str
    worktree_name: str
    worktree_branch: str | None = None
    git_root: str | None = None
    head_commit: str | None = None
    hook_based: bool = False
    created_at: float = 0.0
    creation_duration_ms: int | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorktreeManager:
    """Claude Code style isolated workspace manager for task/subagent runs.

    Git worktrees live under ``<canonical repo>/.claude/worktrees`` and use a
    temporary ``worktree-<slug>`` branch.  Hook-based creation/removal is also
    supported through ``WorktreeCreate``/``WorktreeRemove`` hooks.
    """

    def __init__(self, *, cwd: str | Path | None = None, hook_engine: HookEngine | None = None) -> None:
        self.cwd = Path(cwd or os.getenv("AGENT_WORKSPACE_ROOT") or os.getcwd()).expanduser().resolve()
        self.hook_engine = hook_engine or HookEngine()
        self._active: dict[str, WorktreeInfo] = {}

    async def can_create(self) -> bool:
        return await self._has_worktree_create_hook() or await asyncio.to_thread(_find_git_root, self.cwd)

    async def create_agent_worktree(self, slug: str, *, session_id: str = "") -> WorktreeInfo:
        validate_worktree_slug(slug)
        started = time.monotonic()

        hook_result = await self._run_create_hook(slug, session_id=session_id)
        if hook_result:
            info = WorktreeInfo(
                session_id=session_id,
                workspace_id=slug,
                original_cwd=str(self.cwd),
                worktree_path=hook_result,
                worktree_name=slug,
                hook_based=True,
                created_at=time.time(),
                creation_duration_ms=int((time.monotonic() - started) * 1000),
            )
            self._active[info.worktree_path] = info
            return info

        git_root = await asyncio.to_thread(_find_canonical_git_root, self.cwd)
        if not git_root:
            raise RuntimeError(
                "Cannot create isolated workspace: not in a git repository and no WorktreeCreate hooks are configured."
            )
        info = await asyncio.to_thread(_create_git_worktree, git_root, slug, str(self.cwd))
        info.session_id = session_id
        info.creation_duration_ms = int((time.monotonic() - started) * 1000)
        self._active[info.worktree_path] = info
        return info

    async def remove_agent_worktree(self, info: WorktreeInfo | dict[str, Any], *, force: bool = True) -> bool:
        worktree = _coerce_info(info)
        if worktree.hook_based:
            removed = await self._run_remove_hook(worktree.worktree_path, session_id=worktree.session_id)
            if removed:
                self._active.pop(worktree.worktree_path, None)
            return removed
        if not worktree.git_root:
            return False
        ok = await asyncio.to_thread(_remove_git_worktree, worktree, force)
        if ok:
            self._active.pop(worktree.worktree_path, None)
        return ok

    async def has_changes(self, info: WorktreeInfo | dict[str, Any]) -> bool:
        worktree = _coerce_info(info)
        if worktree.hook_based or not worktree.head_commit:
            return True
        return await asyncio.to_thread(_has_git_changes, worktree.worktree_path, worktree.head_commit)

    async def cleanup_if_clean(self, info: WorktreeInfo | dict[str, Any]) -> tuple[bool, bool]:
        """Return ``(removed, has_changes)``; fail-closed by preserving on uncertainty."""
        changed = await self.has_changes(info)
        if changed:
            return False, True
        return await self.remove_agent_worktree(info), False

    async def shutdown(self) -> None:
        for info in list(self._active.values()):
            with contextlib_suppress_all():
                await self.cleanup_if_clean(info)

    async def _has_worktree_create_hook(self) -> bool:
        return bool(self.hook_engine.get_matching_hooks_for_event(HookEvent.WorktreeCreate)) if hasattr(self.hook_engine, "get_matching_hooks_for_event") else False

    async def _run_create_hook(self, slug: str, *, session_id: str) -> str | None:
        if not await self._has_worktree_create_hook():
            return None
        result = await self.hook_engine.fire(
            HookEvent.WorktreeCreate,
            session_id=session_id,
            cwd=str(self.cwd),
            extra={"name": slug},
        )
        if result.should_block:
            raise RuntimeError(result.stop_reason or "WorktreeCreate hook blocked isolated workspace creation")
        for item in result.results:
            candidate = (item.stdout or "").strip().splitlines()[-1:] or []
            if candidate:
                path = Path(candidate[0]).expanduser().resolve()
                if path.is_dir():
                    return str(path)
        return None

    async def _run_remove_hook(self, worktree_path: str, *, session_id: str) -> bool:
        result = await self.hook_engine.fire(
            HookEvent.WorktreeRemove,
            session_id=session_id,
            cwd=str(self.cwd),
            extra={"worktree_path": worktree_path},
        )
        return bool(result.results) and not result.should_block


def _coerce_info(info: WorktreeInfo | dict[str, Any]) -> WorktreeInfo:
    if isinstance(info, WorktreeInfo):
        return info
    return WorktreeInfo(
        session_id=str(info.get("session_id") or ""),
        workspace_id=str(info.get("workspace_id") or info.get("worktree_name") or ""),
        original_cwd=str(info.get("original_cwd") or info.get("originalCwd") or ""),
        worktree_path=str(info.get("worktree_path") or info.get("worktreePath") or ""),
        worktree_name=str(info.get("worktree_name") or info.get("worktreeName") or info.get("workspace_id") or ""),
        worktree_branch=info.get("worktree_branch") or info.get("worktreeBranch"),
        git_root=info.get("git_root") or info.get("gitRoot"),
        head_commit=info.get("head_commit") or info.get("headCommit"),
        hook_based=bool(info.get("hook_based") or info.get("hookBased")),
        created_at=float(info.get("created_at") or time.time()),
        creation_duration_ms=info.get("creation_duration_ms") or info.get("creationDurationMs"),
        status=str(info.get("status") or "active"),
    )


def _run_git(cwd: Path | str, args: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _find_git_root(cwd: Path) -> bool:
    try:
        return _run_git(cwd, ["rev-parse", "--is-inside-work-tree"], timeout=3).stdout.strip().lower() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_canonical_git_root(cwd: Path) -> Path | None:
    try:
        if _run_git(cwd, ["rev-parse", "--is-inside-work-tree"], timeout=3).stdout.strip().lower() != "true":
            return None
        listed = _run_git(cwd, ["worktree", "list", "--porcelain"], timeout=5)
        if listed.returncode == 0:
            for line in listed.stdout.splitlines():
                if line.startswith("worktree "):
                    return Path(line.removeprefix("worktree ")).expanduser().resolve()
        root = _run_git(cwd, ["rev-parse", "--show-toplevel"], timeout=3)
        if root.returncode == 0 and root.stdout.strip():
            return Path(root.stdout.strip()).expanduser().resolve()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def _worktrees_dir(repo_root: Path) -> Path:
    return repo_root / ".claude" / "worktrees"


def _worktree_path_for(repo_root: Path, slug: str) -> Path:
    return _worktrees_dir(repo_root) / flatten_worktree_slug(slug)


def _create_git_worktree(repo_root: Path, slug: str, original_cwd: str) -> WorktreeInfo:
    branch = worktree_branch_name(slug)
    path = _worktree_path_for(repo_root, slug)
    existing_head = _read_head(path)
    if existing_head:
        os.utime(path, None)
        return WorktreeInfo(
            session_id="",
            workspace_id=slug,
            original_cwd=original_cwd,
            worktree_path=str(path),
            worktree_name=slug,
            worktree_branch=branch,
            git_root=str(repo_root),
            head_commit=existing_head,
            created_at=time.time(),
        )

    _worktrees_dir(repo_root).mkdir(parents=True, exist_ok=True)
    base = _resolve_base_ref(repo_root)
    add = _run_git(repo_root, ["worktree", "add", "-B", branch, str(path), base], timeout=60)
    if add.returncode != 0:
        raise RuntimeError(f"Failed to create isolated workspace: {add.stderr.strip() or add.stdout.strip()}")
    head = _run_git(path, ["rev-parse", "HEAD"], timeout=5)
    head_commit = head.stdout.strip() if head.returncode == 0 else None
    _post_creation_setup(repo_root, path)
    return WorktreeInfo(
        session_id="",
        workspace_id=slug,
        original_cwd=original_cwd,
        worktree_path=str(path),
        worktree_name=slug,
        worktree_branch=branch,
        git_root=str(repo_root),
        head_commit=head_commit,
        created_at=time.time(),
    )


def _resolve_base_ref(repo_root: Path) -> str:
    origin_head = _run_git(repo_root, ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], timeout=3)
    if origin_head.returncode == 0 and origin_head.stdout.strip():
        ref = origin_head.stdout.strip()
        if _run_git(repo_root, ["rev-parse", "--verify", ref], timeout=3).returncode == 0:
            return ref
    current = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"], timeout=3)
    if current.returncode == 0 and current.stdout.strip() and current.stdout.strip() != "HEAD":
        return current.stdout.strip()
    return "HEAD"


def _read_head(worktree_path: Path) -> str | None:
    if not worktree_path.exists():
        return None
    try:
        proc = _run_git(worktree_path, ["rev-parse", "HEAD"], timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def _post_creation_setup(repo_root: Path, worktree_path: Path) -> None:
    local_settings = repo_root / ".claude" / "settings.local.json"
    if local_settings.is_file():
        dest = worktree_path / ".claude" / "settings.local.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_settings, dest)
    include = repo_root / ".worktreeinclude"
    if include.is_file():
        for raw in include.read_text(encoding="utf-8", errors="ignore").splitlines():
            rel = raw.strip()
            if not rel or rel.startswith("#") or ".." in Path(rel).parts or Path(rel).is_absolute():
                continue
            src = repo_root / rel
            dest = worktree_path / rel
            if src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)


def _remove_git_worktree(info: WorktreeInfo, force: bool) -> bool:
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(info.worktree_path)
    remove = _run_git(info.git_root or info.original_cwd, args, timeout=30)
    if remove.returncode != 0:
        return False
    if info.worktree_branch:
        _run_git(info.git_root or info.original_cwd, ["branch", "-D", info.worktree_branch], timeout=10)
    return True


def _has_git_changes(worktree_path: str, head_commit: str) -> bool:
    status = _run_git(worktree_path, ["status", "--porcelain"], timeout=10)
    if status.returncode != 0 or status.stdout.strip():
        return True
    rev = _run_git(worktree_path, ["rev-list", "--count", f"{head_commit}..HEAD"], timeout=10)
    if rev.returncode != 0:
        return True
    try:
        return int(rev.stdout.strip() or "0") > 0
    except ValueError:
        return True


class contextlib_suppress_all:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return True
