from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


TeamMemberStatus = Literal["idle", "working", "shutdown", "failed"]
TEAM_MEMBER_STATUSES: tuple[TeamMemberStatus, ...] = ("idle", "working", "shutdown", "failed")

MessageType = Literal[
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_request",
    "plan_approval_response",
]
VALID_MESSAGE_TYPES: tuple[str, ...] = (
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_request",
    "plan_approval_response",
)


@dataclass(slots=True)
class TeamMessage:
    type: str
    sender: str
    content: str
    timestamp: float = field(default_factory=time.time)
    request_id: str | None = None
    summary: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamMessage":
        return cls(
            type=str(data.get("type") or "message"),
            sender=str(data.get("sender") or data.get("from") or "unknown"),
            content=str(data.get("content") or data.get("text") or ""),
            timestamp=float(data.get("timestamp") or time.time()),
            request_id=str(data.get("request_id") or data.get("requestId")) if data.get("request_id") or data.get("requestId") else None,
            summary=str(data.get("summary")) if data.get("summary") else None,
            extra=data.get("extra") if isinstance(data.get("extra"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.request_id is None:
            payload.pop("request_id", None)
        if self.summary is None:
            payload.pop("summary", None)
        if not self.extra:
            payload.pop("extra", None)
        return payload


@dataclass(slots=True)
class TeamMember:
    name: str
    role: str
    status: TeamMemberStatus = "idle"
    child_session_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_prompt: str = ""
    last_result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamMember":
        status = str(data.get("status") or "idle")
        if status not in TEAM_MEMBER_STATUSES:
            status = "idle"
        return cls(
            name=str(data.get("name") or ""),
            role=str(data.get("role") or "general-purpose"),
            status=status,  # type: ignore[arg-type]
            child_session_id=str(data.get("child_session_id") or data.get("childSessionId") or ""),
            created_at=float(data.get("created_at") or data.get("createdAt") or time.time()),
            updated_at=float(data.get("updated_at") or data.get("updatedAt") or time.time()),
            last_prompt=str(data.get("last_prompt") or data.get("lastPrompt") or ""),
            last_result=str(data.get("last_result") or data.get("lastResult") or ""),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TeamRecord:
    name: str
    description: str = ""
    lead_session_id: str = ""
    user_id: str = ""
    org_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    members: list[TeamMember] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamRecord":
        raw_members = data.get("members") if isinstance(data.get("members"), list) else []
        return cls(
            name=str(data.get("name") or data.get("team_name") or ""),
            description=str(data.get("description") or ""),
            lead_session_id=str(data.get("lead_session_id") or data.get("leadSessionId") or ""),
            user_id=str(data.get("user_id") or data.get("userId") or ""),
            org_id=str(data.get("org_id") or data.get("orgId") or ""),
            created_at=float(data.get("created_at") or data.get("createdAt") or time.time()),
            updated_at=float(data.get("updated_at") or data.get("updatedAt") or time.time()),
            members=[TeamMember.from_dict(item) for item in raw_members if isinstance(item, dict)],
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "lead_session_id": self.lead_session_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "members": [member.to_dict() for member in self.members],
            "metadata": self.metadata,
        }

    def find_member(self, name: str) -> TeamMember | None:
        wanted = sanitize_name(name)
        return next((member for member in self.members if member.name == wanted), None)


class TeamStore:
    """File-backed team, teammate, and mailbox store.

    The storage model mirrors the claude-code-sourcemap team lessons while using
    a safer per-team directory and locked JSONL inboxes:

      ~/.my-agent-core/teams/{team}/team.json
      ~/.my-agent-core/teams/{team}/inboxes/{agent}.jsonl
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        configured = base_dir or os.getenv("AGENT_TEAMS_DIR")
        self._base_dir = Path(configured or Path.home() / ".my-agent-core" / "teams").expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def create_team(
        self,
        *,
        name: str,
        lead_session_id: str,
        description: str = "",
        user_id: str = "",
        org_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TeamRecord:
        team_name = sanitize_name(name)
        if not team_name:
            raise ValueError("team name cannot be empty")
        with self._team_lock(team_name):
            existing = self.get_team(team_name)
            if existing is not None:
                return existing
            team = TeamRecord(
                name=team_name,
                description=description.strip(),
                lead_session_id=lead_session_id,
                user_id=user_id,
                org_id=org_id,
                metadata=metadata or {},
            )
            self._write_team(team)
            self._inbox_dir(team_name).mkdir(parents=True, exist_ok=True)
            return team

    def get_team(self, name: str) -> TeamRecord | None:
        team_name = sanitize_name(name)
        if not team_name:
            return None
        path = self._team_path(team_name)
        if not path.exists():
            return None
        try:
            return TeamRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def require_team(self, name: str) -> TeamRecord:
        team = self.get_team(name)
        if team is None:
            raise KeyError(f"Team '{sanitize_name(name)}' not found")
        return team

    def list_teams(self, *, lead_session_id: str | None = None, user_id: str | None = None) -> list[TeamRecord]:
        teams: list[TeamRecord] = []
        for path in sorted(self._base_dir.glob("*/team.json")):
            with contextlib.suppress(Exception):
                team = TeamRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if lead_session_id is not None and team.lead_session_id != lead_session_id:
                    continue
                if user_id is not None and team.user_id and team.user_id != user_id:
                    continue
                teams.append(team)
        return teams

    def delete_team(self, name: str, *, force: bool = False) -> bool:
        team_name = sanitize_name(name)
        team = self.get_team(team_name)
        if team is None:
            return False
        active = [member.name for member in team.members if member.status in {"working", "idle"}]
        if active and not force:
            raise RuntimeError(f"team has active members: {', '.join(active)}")
        with self._team_lock(team_name):
            team_dir = self._team_dir(team_name)
            if not team_dir.exists():
                return False
            for child in sorted(team_dir.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            team_dir.rmdir()
            return True

    def upsert_member(
        self,
        *,
        team_name: str,
        member_name: str,
        role: str,
        child_session_id: str,
        status: TeamMemberStatus = "idle",
        prompt: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TeamMember:
        team_key = sanitize_name(team_name)
        member_key = sanitize_name(member_name)
        if not member_key:
            raise ValueError("member name cannot be empty")
        with self._team_lock(team_key):
            team = self.require_team(team_key)
            member = team.find_member(member_key)
            now = time.time()
            if member is None:
                member = TeamMember(
                    name=member_key,
                    role=role.strip() or "general-purpose",
                    status=status,
                    child_session_id=child_session_id,
                    last_prompt=prompt,
                    metadata=metadata or {},
                )
                team.members.append(member)
            else:
                member.role = role.strip() or member.role
                member.status = status
                member.child_session_id = child_session_id or member.child_session_id
                member.updated_at = now
                if prompt:
                    member.last_prompt = prompt
                if metadata:
                    member.metadata.update(metadata)
            team.updated_at = now
            self._write_team(team)
            return member

    def update_member(
        self,
        *,
        team_name: str,
        member_name: str,
        status: TeamMemberStatus | None = None,
        last_result: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TeamMember | None:
        team_key = sanitize_name(team_name)
        member_key = sanitize_name(member_name)
        with self._team_lock(team_key):
            team = self.get_team(team_key)
            if team is None:
                return None
            member = team.find_member(member_key)
            if member is None:
                return None
            if status is not None:
                member.status = status
            if last_result is not None:
                member.last_result = last_result
            if metadata:
                member.metadata.update(metadata)
            member.updated_at = time.time()
            team.updated_at = member.updated_at
            self._write_team(team)
            return member

    def send_message(
        self,
        *,
        team_name: str,
        sender: str,
        recipient: str,
        content: str,
        msg_type: str = "message",
        request_id: str | None = None,
        summary: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> list[str]:
        team = self.require_team(team_name)
        if msg_type not in VALID_MESSAGE_TYPES:
            raise ValueError(f"invalid message type: {msg_type}")
        recipients: list[str]
        if recipient == "*":
            if msg_type not in {"message", "broadcast"}:
                raise ValueError("structured protocol messages cannot be broadcast")
            recipients = [member.name for member in team.members if member.name != sanitize_name(sender)]
            msg_type = "broadcast"
        else:
            recipients = [sanitize_name(recipient)]
        sent: list[str] = []
        for target in recipients:
            if not target:
                continue
            message = TeamMessage(
                type=msg_type,
                sender=sanitize_name(sender) or sender,
                content=content,
                request_id=request_id,
                summary=summary,
                extra=extra or {},
            )
            self._append_inbox(team.name, target, message)
            sent.append(target)
        return sent

    def read_inbox(self, *, team_name: str, recipient: str, drain: bool = True) -> list[TeamMessage]:
        team_key = sanitize_name(team_name)
        recipient_key = sanitize_name(recipient)
        path = self._inbox_path(team_key, recipient_key)
        if not path.exists():
            return []
        with self._file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            messages = [TeamMessage.from_dict(json.loads(line)) for line in lines if line.strip()]
            if drain:
                _atomic_write_text(path, "")
            return messages

    def render_context(self, *, session_id: str, user_id: str = "") -> str:
        teams = self.list_teams(lead_session_id=session_id, user_id=user_id or None)
        if not teams:
            return ""
        lines = ["# Agent Team Context", "", "Persistent named teammates are available via Agent/SendMessage/ReadInbox tools."]
        for team in teams:
            lines.append(f"- Team `{team.name}`: {team.description or 'no description'}")
            if not team.members:
                lines.append("  - No teammates yet.")
            for member in team.members:
                lines.append(
                    f"  - {member.name} ({member.role}) status={member.status} session={member.child_session_id or 'unassigned'}"
                )
        return "\n".join(lines)

    def _team_dir(self, team_name: str) -> Path:
        return self._base_dir / sanitize_name(team_name)

    def _team_path(self, team_name: str) -> Path:
        return self._team_dir(team_name) / "team.json"

    def _inbox_dir(self, team_name: str) -> Path:
        return self._team_dir(team_name) / "inboxes"

    def _inbox_path(self, team_name: str, recipient: str) -> Path:
        return self._inbox_dir(team_name) / f"{sanitize_name(recipient)}.jsonl"

    def _write_team(self, team: TeamRecord) -> None:
        path = self._team_path(team.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(team.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))

    def _append_inbox(self, team_name: str, recipient: str, message: TeamMessage) -> None:
        path = self._inbox_path(team_name, recipient)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    @contextlib.contextmanager
    def _team_lock(self, team_name: str):
        team_dir = self._team_dir(team_name)
        team_dir.mkdir(parents=True, exist_ok=True)
        with self._file_lock(team_dir / ".lock"):
            yield

    @contextlib.contextmanager
    def _file_lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", str(name or "").strip()).strip(".-_")
    return cleaned[:80]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
