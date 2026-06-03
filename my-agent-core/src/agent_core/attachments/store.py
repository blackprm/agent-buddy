from __future__ import annotations

import hashlib
import mimetypes
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_core.sqlite_utils import connect_sqlite


ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/mpeg": ".mpeg",
    "video/x-msvideo": ".avi",
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}

ALLOWED_MEDIA_TYPES = {**ALLOWED_IMAGE_TYPES, **ALLOWED_VIDEO_TYPES, **ALLOWED_AUDIO_TYPES}


def _detect_media_type(data: bytes) -> str:
    try:
        import filetype
    except ImportError as exc:
        raise ValueError("media type detection requires the 'filetype' package") from exc
    kind = filetype.guess(data)
    return str(getattr(kind, "mime", "") or "") if kind else ""


@dataclass(slots=True)
class ImageAttachment:
    id: str
    user_id: str
    org_id: str
    session_id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    path: str
    created_at: float
    metadata: dict[str, Any]

    def to_dict(self, *, include_path: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_path:
            data.pop("path", None)
        return data


class ImageAttachmentStore:
    """Session-scoped image/video attachment storage.

    Metadata is stored in SQLite while image bytes are stored on disk.  Tools
    must resolve attachments through ``get_authorized`` so a user/session can
    only read its own uploaded files.
    """

    def __init__(self, root_dir: str | Path, *, max_size_bytes: int = 10 * 1024 * 1024) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.files_dir = self.root_dir / "files"
        self.db_path = self.root_dir / "attachments.sqlite3"
        self.max_size_bytes = max_size_bytes
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, row_factory=sqlite3.Row)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_attachments (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_attachments_session ON image_attachments(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_image_attachments_owner ON image_attachments(user_id, org_id)")

    def save_image(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str | None,
        user_id: str,
        org_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ImageAttachment:
        return self.save_media(
            data=data,
            filename=filename,
            content_type=content_type,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            metadata=metadata,
        )

    def save_media(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str | None,
        user_id: str,
        org_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ImageAttachment:
        import json

        if not data:
            raise ValueError("media attachment is empty")
        if len(data) > self.max_size_bytes:
            raise ValueError(f"media attachment exceeds max size of {self.max_size_bytes} bytes")

        raw_content_type = (content_type or "").split(";", 1)[0].strip().lower()
        guessed_type = raw_content_type
        should_detect = not guessed_type or guessed_type == "application/octet-stream"
        if not guessed_type or guessed_type == "application/octet-stream":
            guessed_type = mimetypes.guess_type(filename or "")[0] or ""
        if guessed_type not in ALLOWED_MEDIA_TYPES and should_detect:
            guessed_type = _detect_media_type(data)
        if guessed_type not in ALLOWED_MEDIA_TYPES:
            raise ValueError(f"unsupported media content type: {content_type or 'unknown'}")

        if guessed_type.startswith("image/"):
            prefix = "img"
        elif guessed_type.startswith("video/"):
            prefix = "vid"
        else:
            prefix = "aud"
        attachment_id = f"{prefix}_{uuid.uuid4().hex}"
        safe_filename = Path(filename or "media").name or "media"
        ext = ALLOWED_MEDIA_TYPES[guessed_type]
        digest = hashlib.sha256(data).hexdigest()
        rel_path = Path(org_id) / user_id / session_id / f"{attachment_id}{ext}"
        abs_path = (self.files_dir / rel_path).resolve()
        if not str(abs_path).startswith(str(self.files_dir)):
            raise ValueError("invalid attachment path")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)

        item = ImageAttachment(
            id=attachment_id,
            user_id=str(user_id),
            org_id=str(org_id),
            session_id=str(session_id),
            filename=safe_filename,
            content_type=guessed_type,
            size_bytes=len(data),
            sha256=digest,
            path=str(abs_path),
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_attachments
                    (id, user_id, org_id, session_id, filename, content_type, size_bytes, sha256, path, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.user_id,
                    item.org_id,
                    item.session_id,
                    item.filename,
                    item.content_type,
                    item.size_bytes,
                    item.sha256,
                    item.path,
                    item.created_at,
                    json.dumps(item.metadata, ensure_ascii=False),
                ),
            )
        return item

    def get(self, attachment_id: str) -> ImageAttachment | None:
        import json

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM image_attachments WHERE id = ?", (attachment_id,)).fetchone()
        if row is None:
            return None
        return ImageAttachment(
            id=row["id"],
            user_id=row["user_id"],
            org_id=row["org_id"],
            session_id=row["session_id"],
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=int(row["size_bytes"]),
            sha256=row["sha256"],
            path=row["path"],
            created_at=float(row["created_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get_authorized(self, *, attachment_id: str, user_id: str, org_id: str, session_id: str) -> ImageAttachment | None:
        item = self.get(attachment_id)
        if item is None:
            return None
        if item.user_id != str(user_id) or item.org_id != str(org_id) or item.session_id != str(session_id):
            return None
        return item

    def read_bytes(self, item: ImageAttachment) -> bytes:
        path = Path(item.path).resolve()
        if not str(path).startswith(str(self.files_dir)):
            raise ValueError("invalid attachment path")
        return path.read_bytes()
