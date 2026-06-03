from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(slots=True)
class TosConfig:
    ak: str
    sk: str
    bucket: str
    endpoint: str = "tos-cn-beijing.volces.com"
    region: str = "cn-beijing"
    expires: int = 86400

    @classmethod
    def from_env(cls) -> "TosConfig | None":
        ak = os.getenv("AGENT_TOS_AK") or os.getenv("TOS_AK")
        sk = os.getenv("AGENT_TOS_SK") or os.getenv("TOS_SK")
        bucket = os.getenv("AGENT_TOS_BUCKET") or os.getenv("TOS_BUCKET")
        if not ak or not sk or not bucket:
            return None
        return cls(
            ak=ak,
            sk=sk,
            bucket=bucket,
            endpoint=os.getenv("AGENT_TOS_ENDPOINT") or os.getenv("TOS_ENDPOINT") or "tos-cn-beijing.volces.com",
            region=os.getenv("AGENT_TOS_REGION") or os.getenv("TOS_REGION") or "cn-beijing",
            expires=int(os.getenv("AGENT_TOS_SIGN_EXPIRES") or os.getenv("TOS_SIGN_EXPIRES") or "86400"),
        )


@dataclass(slots=True)
class UploadedTosObject:
    key: str
    stored_url: str
    signed_url: str
    expires: int


class TosMediaStore:
    """Small TOS bridge for media references passed to video generation.

    Local attachments are uploaded to TOS and passed to the video API as a
    temporary signed HTTPS URL. Metadata keeps the stable ``tos://bucket/key``
    form, mirroring seedance_vefaas' asset storage pattern.
    """

    def __init__(self, config: TosConfig | None = None) -> None:
        config = config or TosConfig.from_env()
        if config is None:
            raise ValueError("TOS is not configured. Set TOS_AK, TOS_SK and TOS_BUCKET (or AGENT_TOS_* equivalents).")
        self.config = config
        self._client: Any | None = None
        self._tos_module: Any | None = None

    @property
    def bucket(self) -> str:
        return self.config.bucket

    def _client_for_thread(self) -> tuple[Any, Any]:
        if self._client is None:
            try:
                import tos  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - environment guard
                raise ImportError("TOS upload requires the 'tos' package. Install my-agent-core[web] or add tos>=2.6.") from exc
            self._tos_module = tos
            self._client = tos.TosClientV2(
                self.config.ak,
                self.config.sk,
                self.config.endpoint,
                self.config.region,
                max_retry_count=2,
            )
        return self._client, self._tos_module

    async def upload_bytes(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str,
        user_id: str,
        org_id: str,
        session_id: str,
        media_kind: str,
        expires: int | None = None,
    ) -> UploadedTosObject:
        if not data:
            raise ValueError("cannot upload empty media attachment to TOS")
        key = make_asset_tos_key(
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            media_kind=media_kind,
            filename=filename,
            content_type=content_type,
        )
        await asyncio.to_thread(self._put_object, key, data, content_type)
        signed_url = await self.presign_key(key, expires=expires)
        ttl = int(expires or self.config.expires)
        return UploadedTosObject(key=key, stored_url=f"tos://{self.config.bucket}/{key}", signed_url=signed_url, expires=ttl)

    async def presign_key(self, key: str, *, expires: int | None = None) -> str:
        return await asyncio.to_thread(self._presign_key, key, int(expires or self.config.expires))

    def parse_tos_uri(self, value: str) -> str | None:
        parsed = urlparse(str(value or ""))
        if parsed.scheme != "tos":
            return None
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError("invalid TOS URL: missing key")
        if bucket and bucket != self.config.bucket:
            raise ValueError(f"TOS bucket mismatch: expected {self.config.bucket}, got {bucket}")
        return key

    def _put_object(self, key: str, data: bytes, content_type: str) -> None:
        client, _ = self._client_for_thread()
        client.put_object(
            self.config.bucket,
            key,
            content=data,
            content_length=len(data),
            content_type=content_type or "application/octet-stream",
        )

    def _presign_key(self, key: str, expires: int) -> str:
        client, tos = self._client_for_thread()
        out = client.pre_signed_url(tos.HttpMethodType.Http_Method_Get, self.config.bucket, key, expires=expires)
        signed_url = str(getattr(out, "signed_url", "") or "")
        if not signed_url:
            raise ValueError("TOS pre_signed_url did not return signed_url")
        return signed_url


def make_asset_tos_key(*, org_id: str, user_id: str, session_id: str, media_kind: str, filename: str, content_type: str) -> str:
    today = time.strftime("%Y-%m-%d")
    ext = Path(filename or "").suffix.lower()
    if not ext:
        ext = _extension_for_content_type(content_type)
    kind = _slug(media_kind or "media", default="media")
    org = _slug(org_id or "org", default="org")
    user = _slug(user_id or "user", default="user")
    session = _slug(session_id or "session", default="session")
    return f"{org}/{user}/{session}/assets/{today}/{kind}_{uuid.uuid4().hex}{ext}"


def is_tos_configured() -> bool:
    return TosConfig.from_env() is not None


def _slug(value: str, *, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value or "")).strip(".-/")
    return cleaned[:96] or default


def _extension_for_content_type(content_type: str) -> str:
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/mpeg": ".mpeg",
        "video/x-msvideo": ".avi",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
    }
    return mapping.get((content_type or "").split(";", 1)[0].lower(), ".bin")
