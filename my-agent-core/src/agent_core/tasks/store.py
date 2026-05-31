from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskStatus = Literal["pending", "in_progress", "completed"]
TASK_STATUSES: tuple[TaskStatus, ...] = ("pending", "in_progress", "completed")
_TASK_ID_RE = re.compile(r"^\d+$")
_HIGH_WATER_MARK_FILE = ".highwatermark"
_LOCK_FILE = ".lock"


@dataclass(slots=True)
class TaskRecord:
    """Claude Code style persistent task record.

    The real CC record uses numeric string IDs and the fields below:
    id/subject/description/activeForm/owner/status/blocks/blockedBy/metadata.
    """

    id: str
    subject: str
    description: str
    status: TaskStatus = "pending"
    blocks: list[str] = field(default_factory=list)
    blockedBy: list[str] = field(default_factory=list)
    activeForm: str | None = None
    owner: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        status = data.get("status")
        if status == "open":
            status = "pending"
        elif status == "resolved":
            status = "completed"
        elif status in {"planning", "implementing", "reviewing", "verifying"}:
            status = "in_progress"
        if status not in TASK_STATUSES:
            status = "pending"
        return cls(
            id=str(data.get("id") or ""),
            subject=str(data.get("subject") or ""),
            description=str(data.get("description") or ""),
            status=status,  # type: ignore[arg-type]
            blocks=_task_id_list(data.get("blocks")),
            blockedBy=_task_id_list(data.get("blockedBy")),
            activeForm=str(data["activeForm"]) if data.get("activeForm") else None,
            owner=str(data["owner"]) if data.get("owner") else None,
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClaimTaskResult:
    success: bool
    reason: str | None = None
    task: TaskRecord | None = None
    busyWithTasks: list[str] = field(default_factory=list)
    blockedByTasks: list[str] = field(default_factory=list)


class FileTaskStore:
    """File-backed task list matching Claude Code's `utils/tasks.ts` semantics.

    Storage shape:
      ~/.my-agent-core/tasks/{task_list_id}/{id}.json
      ~/.my-agent-core/tasks/{task_list_id}/.highwatermark
      ~/.my-agent-core/tasks/{task_list_id}/.lock

    `AGENT_TASKS_DIR` may override the base tasks directory. `task_list_id`
    defaults to the current session when tools are used, mirroring CC's
    getTaskListId fallback to session ID.
    """

    def __init__(self, base_dir: str | Path | None = None, *, task_list_id: str = "tasklist") -> None:
        configured = base_dir or os.getenv("AGENT_TASKS_DIR")
        self._base_dir = Path(configured or Path.home() / ".my-agent-core" / "tasks").expanduser().resolve()
        self._task_list_id = sanitize_path_component(task_list_id or "tasklist")
        self._dir = self._base_dir / self._task_list_id
        self.ensure_dir()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def task_list_id(self) -> str:
        return self._task_list_id

    @property
    def tasks_dir(self) -> Path:
        return self._dir

    def for_task_list(self, task_list_id: str) -> "FileTaskStore":
        return FileTaskStore(self._base_dir, task_list_id=task_list_id)

    def ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / _LOCK_FILE).touch(exist_ok=True)

    def reset(self) -> None:
        with self._list_lock():
            highest = max(self._highest_from_files(), self._read_highwatermark())
            if highest > 0:
                self._write_highwatermark(highest)
            for path in self._dir.glob("*.json"):
                if not path.name.startswith("."):
                    with contextlib.suppress(OSError):
                        path.unlink()

    def create(
        self,
        *,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        subject = subject.strip()
        if not subject:
            raise ValueError("subject cannot be empty")
        with self._list_lock():
            task_id = str(max(self._highest_from_files(), self._read_highwatermark()) + 1)
            task = TaskRecord(
                id=task_id,
                subject=subject,
                description=description,
                activeForm=active_form.strip() if isinstance(active_form, str) and active_form.strip() else None,
                status="pending",
                owner=None,
                blocks=[],
                blockedBy=[],
                metadata=metadata or {},
            )
            self._write_task(task)
            self._write_highwatermark(int(task_id))
            return task

    def get(self, task_id: str) -> TaskRecord | None:
        task_id = normalize_task_id(task_id)
        path = self._task_path(task_id)
        if not path.exists():
            return None
        try:
            return TaskRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def require(self, task_id: str) -> TaskRecord:
        task = self.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        return task

    def list(self, *, include_internal: bool = True) -> list[TaskRecord]:
        tasks = [task for task_id in self._task_ids_from_files() if (task := self.get(task_id)) is not None]
        if not include_internal:
            tasks = [task for task in tasks if not task.metadata.get("_internal")]
        return tasks

    def update(
        self,
        task_id: str,
        *,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        status: TaskStatus | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord | None:
        task_id = normalize_task_id(task_id)
        path = self._task_path(task_id)
        if not path.exists():
            return None
        with self._file_lock(path):
            task = self.get(task_id)
            if task is None:
                return None
            if subject is not None:
                if not subject.strip():
                    raise ValueError("subject cannot be empty")
                task.subject = subject.strip()
            if description is not None:
                task.description = description
            if active_form is not None:
                task.activeForm = active_form.strip() or None
            if status is not None:
                if status not in TASK_STATUSES:
                    raise ValueError(f"invalid status: {status}")
                task.status = status
            if owner is not None:
                task.owner = owner.strip() or None
            if metadata is not None:
                merged = dict(task.metadata)
                for key, value in metadata.items():
                    if value is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = value
                task.metadata = merged
            self._write_task(task)
            return task

    def delete(self, task_id: str) -> bool:
        task_id = normalize_task_id(task_id)
        with self._list_lock():
            numeric = int(task_id)
            if numeric > self._read_highwatermark():
                self._write_highwatermark(numeric)
            path = self._task_path(task_id)
            if not path.exists():
                return False
            path.unlink()
            for task in self.list():
                new_blocks = [item for item in task.blocks if item != task_id]
                new_blocked_by = [item for item in task.blockedBy if item != task_id]
                if new_blocks != task.blocks or new_blocked_by != task.blockedBy:
                    task.blocks = new_blocks
                    task.blockedBy = new_blocked_by
                    self._write_task(task)
            return True

    def block_task(self, from_task_id: str, to_task_id: str) -> bool:
        from_task_id = normalize_task_id(from_task_id)
        to_task_id = normalize_task_id(to_task_id)
        if from_task_id == to_task_id:
            raise ValueError("task cannot block itself")
        with self._list_lock():
            from_task = self.get(from_task_id)
            to_task = self.get(to_task_id)
            if not from_task or not to_task:
                return False
            if self._would_create_cycle(from_task_id, to_task_id):
                raise ValueError(f"dependency cycle detected: {from_task_id} -> {to_task_id}")
            if to_task_id not in from_task.blocks:
                from_task.blocks.append(to_task_id)
                from_task.blocks.sort(key=int)
                self._write_task(from_task)
            if from_task_id not in to_task.blockedBy:
                to_task.blockedBy.append(from_task_id)
                to_task.blockedBy.sort(key=int)
                self._write_task(to_task)
            return True

    def claim(self, task_id: str, claimant_agent_id: str, *, check_agent_busy: bool = False) -> ClaimTaskResult:
        task_id = normalize_task_id(task_id)
        if check_agent_busy:
            with self._list_lock():
                return self._claim_locked(task_id, claimant_agent_id, check_agent_busy=True)
        path = self._task_path(task_id)
        if not path.exists():
            return ClaimTaskResult(success=False, reason="task_not_found")
        with self._file_lock(path):
            return self._claim_locked(task_id, claimant_agent_id, check_agent_busy=False)

    def unassign_owner(self, owner: str) -> list[TaskRecord]:
        unassigned: list[TaskRecord] = []
        with self._list_lock():
            for task in self.list():
                if task.status != "completed" and task.owner == owner:
                    task.owner = None
                    task.status = "pending"
                    self._write_task(task)
                    unassigned.append(task)
        return unassigned

    def available_tasks(self) -> list[TaskRecord]:
        tasks = self.list(include_internal=False)
        unresolved = {task.id for task in tasks if task.status != "completed"}
        return [
            task
            for task in tasks
            if task.status == "pending" and not task.owner and all(dep not in unresolved for dep in task.blockedBy)
        ]

    def _claim_locked(self, task_id: str, claimant_agent_id: str, *, check_agent_busy: bool) -> ClaimTaskResult:
        all_tasks = self.list()
        task = next((item for item in all_tasks if item.id == task_id), None)
        if task is None:
            return ClaimTaskResult(success=False, reason="task_not_found")
        if task.owner and task.owner != claimant_agent_id:
            return ClaimTaskResult(success=False, reason="already_claimed", task=task)
        if task.status == "completed":
            return ClaimTaskResult(success=False, reason="already_resolved", task=task)
        unresolved = {item.id for item in all_tasks if item.status != "completed"}
        blocked_by = [dep for dep in task.blockedBy if dep in unresolved]
        if blocked_by:
            return ClaimTaskResult(success=False, reason="blocked", task=task, blockedByTasks=blocked_by)
        if check_agent_busy:
            busy = [item.id for item in all_tasks if item.status != "completed" and item.owner == claimant_agent_id and item.id != task_id]
            if busy:
                return ClaimTaskResult(success=False, reason="agent_busy", task=task, busyWithTasks=busy)
        task.owner = claimant_agent_id
        self._write_task(task)
        return ClaimTaskResult(success=True, task=task)

    def _would_create_cycle(self, from_task_id: str, to_task_id: str) -> bool:
        # Adding from -> to is cyclic if to already reaches from through blocks.
        def reaches(current: str, target: str, seen: set[str]) -> bool:
            if current == target:
                return True
            if current in seen:
                return False
            seen.add(current)
            task = self.get(current)
            if not task:
                return False
            return any(reaches(next_id, target, seen) for next_id in task.blocks)

        return reaches(to_task_id, from_task_id, set())

    def _task_path(self, task_id: str) -> Path:
        return self._dir / f"{normalize_task_id(task_id)}.json"

    def _task_ids_from_files(self) -> list[str]:
        ids = [path.stem for path in self._dir.glob("*.json") if _TASK_ID_RE.match(path.stem)]
        return sorted(ids, key=int)

    def _highest_from_files(self) -> int:
        return max([int(task_id) for task_id in self._task_ids_from_files()], default=0)

    def _read_highwatermark(self) -> int:
        try:
            return int((self._dir / _HIGH_WATER_MARK_FILE).read_text(encoding="utf-8").strip() or "0")
        except Exception:
            return 0

    def _write_highwatermark(self, value: int) -> None:
        _atomic_write_text(self._dir / _HIGH_WATER_MARK_FILE, str(value))

    def _write_task(self, task: TaskRecord) -> None:
        if not _TASK_ID_RE.match(task.id):
            raise ValueError(f"invalid task id: {task.id}")
        _atomic_write_text(self._task_path(task.id), json.dumps(task.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))

    @contextlib.contextmanager
    def _list_lock(self):
        self.ensure_dir()
        with _locked_file(self._dir / _LOCK_FILE):
            yield

    @contextlib.contextmanager
    def _file_lock(self, path: Path):
        with _locked_file(path):
            yield


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", value or "tasklist")
    return sanitized or "tasklist"


def normalize_task_id(task_id: str | int) -> str:
    normalized = str(task_id).strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if normalized.startswith("task_"):
        normalized = normalized.split("_", 1)[1]
    if not _TASK_ID_RE.match(normalized):
        raise ValueError(f"invalid task id: {task_id!r}; expected numeric string")
    return normalized


def _task_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        with contextlib.suppress(ValueError):
            task_id = normalize_task_id(item)
            if task_id not in ids:
                ids.append(task_id)
    return ids


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with open(path, "r+", encoding="utf-8") as fh:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(Exception):
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
