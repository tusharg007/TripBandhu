"""Ownership-bound, versioned approved-itinerary artifacts.

The primary implementation stores approved documents and PDF bytes in
PostgreSQL.  Development mode can fall back to memory, but it deliberately
reports that persistence is unavailable after a process restart.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from itinerary_document import ItineraryDocument, ItineraryValidationError, require_exportable


SESSION_COOKIE_NAME = "tripbandhu_session"
_SIGNING_SECRET = os.getenv("SESSION_SIGNING_SECRET") or secrets.token_urlsafe(48)


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def sign_session_id(session_id: str) -> str:
    payload = str(session_id or "").encode("utf-8")
    signature = hmac.new(_SIGNING_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{session_id}.{signature}"


def verify_signed_session(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    session_id, signature = value.rsplit(".", 1)
    expected = hmac.new(_SIGNING_SECRET.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    return session_id if session_id else None


def owner_hash(session_id: str) -> str:
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()


@dataclass
class StoredPlan:
    plan_id: str
    owner_hash: str
    thread_id: str
    version: int
    content_hash: str
    document: ItineraryDocument
    pdf_bytes: bytes | None = None
    pdf_renderer_version: str | None = None
    asset_version: str | None = None
    created_at: str = ""

    def public_reference(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_version": self.version,
            "plan_content_hash": self.content_hash,
            "export_status": "READY" if self.pdf_bytes else "PENDING",
        }


class ArtifactStore:
    """Ownership-bound itinerary artifacts with PostgreSQL and safe memory fallback.

    The database implementation keeps approval and rendered bytes available after
    a process restart. Memory mode exists only for local development or a
    temporarily unavailable database and reports itself as non-durable.
    """

    def __init__(self, database_url: str = "", request_limit: int = 8, request_window_seconds: int = 3600) -> None:
        self._plans: dict[str, StoredPlan] = {}
        self._thread_owners: dict[str, str] = {}
        self._request_windows: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._connection: AsyncConnection | None = None
        self._database_url = str(database_url or "").strip()
        self._request_limit = max(1, int(request_limit))
        self._request_window_seconds = max(1, int(request_window_seconds))
        self.backend = "memory"
        self.persistent = False

    async def initialize(self) -> None:
        """Use PostgreSQL when configured and reachable without blocking startup."""
        if not self._database_url:
            return
        try:
            self._connection = await asyncio.wait_for(
                AsyncConnection.connect(self._database_url, autocommit=True, row_factory=dict_row),
                timeout=5,
            )
            await self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tripbandhu_artifact_threads (
                    thread_id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            await self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tripbandhu_itinerary_artifacts (
                    plan_id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    document_json JSONB NOT NULL,
                    pdf_bytes BYTEA,
                    pdf_renderer_version TEXT,
                    asset_version TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (thread_id, version)
                );
                """
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS tripbandhu_artifact_owner_idx "
                "ON tripbandhu_itinerary_artifacts (owner_hash, plan_id, version);"
            )
            self.backend = "postgresql"
            self.persistent = True
        except Exception as exc:
            self._connection = None
            self.backend = "memory"
            self.persistent = False
            print(f"[artifact_store] PostgreSQL artifacts unavailable ({type(exc).__name__}); using memory.", flush=True)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @staticmethod
    def _document_from_row(row: dict[str, Any]) -> StoredPlan:
        raw_document = row["document_json"]
        if isinstance(raw_document, str):
            raw_document = json.loads(raw_document)
        return StoredPlan(
            plan_id=row["plan_id"],
            owner_hash=row["owner_hash"],
            thread_id=row["thread_id"],
            version=int(row["version"]),
            content_hash=row["content_hash"],
            document=ItineraryDocument.model_validate(raw_document),
            pdf_bytes=bytes(row["pdf_bytes"]) if row.get("pdf_bytes") else None,
            pdf_renderer_version=row.get("pdf_renderer_version"),
            asset_version=row.get("asset_version"),
            created_at=str(row.get("created_at") or ""),
        )

    async def bind_thread(self, session_id: str, thread_id: str) -> None:
        if self._connection is not None:
            await self._connection.execute(
                """
                INSERT INTO tripbandhu_artifact_threads (thread_id, owner_hash)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET updated_at = NOW()
                WHERE tripbandhu_artifact_threads.owner_hash = EXCLUDED.owner_hash;
                """,
                (thread_id, owner_hash(session_id)),
            )
            return
        async with self._lock:
            self._thread_owners[thread_id] = owner_hash(session_id)

    async def owns_thread(self, session_id: str, thread_id: str) -> bool:
        if self._connection is not None:
            async with self._connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT owner_hash FROM tripbandhu_artifact_threads WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cursor.fetchone()
            return bool(row and hmac.compare_digest(row["owner_hash"], owner_hash(session_id)))
        async with self._lock:
            return hmac.compare_digest(self._thread_owners.get(thread_id, ""), owner_hash(session_id))

    async def allow_request(self, session_id: str) -> tuple[bool, int]:
        """Small per-instance abuse guard; provider adapters remain quota-aware."""
        now = time.monotonic()
        key = owner_hash(session_id)
        async with self._lock:
            window = self._request_windows.setdefault(key, deque())
            cutoff = now - self._request_window_seconds
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= self._request_limit:
                retry_after = max(1, int(self._request_window_seconds - (now - window[0])))
                return False, retry_after
            window.append(now)
            return True, 0

    async def save_approved(self, session_id: str, document: ItineraryDocument) -> StoredPlan:
        require_exportable(document)
        record = StoredPlan(
            plan_id=document.plan_id,
            owner_hash=owner_hash(session_id),
            thread_id=document.thread_id,
            version=document.version,
            content_hash=document.content_hash,
            document=document,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        if self._connection is not None:
            existing = await self._get_owned_thread_version(session_id, document.thread_id, document.version)
            if existing is not None:
                if not hmac.compare_digest(existing.content_hash, document.content_hash):
                    raise ItineraryValidationError("An immutable approved itinerary already exists for this review version.")
                return existing
            payload = Jsonb(document.model_dump(mode="json"))
            await self._connection.execute(
                """
                INSERT INTO tripbandhu_itinerary_artifacts
                    (plan_id, owner_hash, thread_id, version, content_hash, document_json)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (plan_id) DO NOTHING;
                """,
                (record.plan_id, record.owner_hash, record.thread_id, record.version, record.content_hash, payload),
            )
            await self.bind_thread(session_id, document.thread_id)
            stored = await self.get_owned_plan(session_id, document.plan_id, document.version)
            if stored is None:
                raise PermissionError("The approved itinerary could not be saved for this session.")
            return stored
        async with self._lock:
            for existing in self._plans.values():
                if existing.thread_id == document.thread_id and existing.version == document.version and hmac.compare_digest(existing.owner_hash, record.owner_hash):
                    if not hmac.compare_digest(existing.content_hash, document.content_hash):
                        raise ItineraryValidationError("An immutable approved itinerary already exists for this review version.")
                    return existing
            existing = self._plans.get(document.plan_id)
            if existing and existing.content_hash == document.content_hash and existing.owner_hash == record.owner_hash:
                return existing
            self._plans[document.plan_id] = record
            self._thread_owners[document.thread_id] = record.owner_hash
        return record

    async def _get_owned_thread_version(self, session_id: str, thread_id: str, version: int) -> StoredPlan | None:
        if self._connection is None:
            return None
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT plan_id, owner_hash, thread_id, version, content_hash, document_json,
                       pdf_bytes, pdf_renderer_version, asset_version, created_at
                FROM tripbandhu_itinerary_artifacts
                WHERE thread_id = %s AND version = %s AND owner_hash = %s
                """,
                (thread_id, version, owner_hash(session_id)),
            )
            row = await cursor.fetchone()
        return self._document_from_row(row) if row else None

    async def get_owned_plan(self, session_id: str, plan_id: str, version: int) -> StoredPlan | None:
        if self._connection is not None:
            async with self._connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT plan_id, owner_hash, thread_id, version, content_hash, document_json,
                           pdf_bytes, pdf_renderer_version, asset_version, created_at
                    FROM tripbandhu_itinerary_artifacts
                    WHERE plan_id = %s AND version = %s AND owner_hash = %s
                    """,
                    (plan_id, version, owner_hash(session_id)),
                )
                row = await cursor.fetchone()
            return self._document_from_row(row) if row else None
        async with self._lock:
            record = self._plans.get(plan_id)
            if not record or record.version != version:
                return None
            if not hmac.compare_digest(record.owner_hash, owner_hash(session_id)):
                return None
            return record

    async def get_or_create_pdf(
        self,
        session_id: str,
        plan_id: str,
        version: int,
        renderer_version: str,
        asset_version: str,
        renderer: Callable[[ItineraryDocument], bytes],
    ) -> tuple[StoredPlan, bytes]:
        record = await self.get_owned_plan(session_id, plan_id, version)
        if record is None:
            raise PermissionError("The requested approved itinerary is unavailable for this session.")
        if record.pdf_bytes and record.pdf_renderer_version == renderer_version and record.asset_version == asset_version:
            return record, record.pdf_bytes

        # Rendering does not call an LLM or provider. Use a per-store lock so
        # duplicate clicks cannot generate a second artifact simultaneously.
        async with self._lock:
            if self._connection is not None:
                record = await self.get_owned_plan(session_id, plan_id, version)
            else:
                record = self._plans.get(plan_id)
            if record is None or not hmac.compare_digest(record.owner_hash, owner_hash(session_id)):
                raise PermissionError("The requested approved itinerary is unavailable for this session.")
            if record.pdf_bytes and record.pdf_renderer_version == renderer_version and record.asset_version == asset_version:
                return record, record.pdf_bytes
            pdf_bytes = await asyncio.to_thread(renderer, record.document)
            record.pdf_bytes = pdf_bytes
            record.pdf_renderer_version = renderer_version
            record.asset_version = asset_version
            if self._connection is not None:
                await self._connection.execute(
                    """
                    UPDATE tripbandhu_itinerary_artifacts
                    SET pdf_bytes = %s, pdf_renderer_version = %s, asset_version = %s
                    WHERE plan_id = %s AND version = %s AND owner_hash = %s
                    """,
                    (pdf_bytes, renderer_version, asset_version, plan_id, version, owner_hash(session_id)),
                )
            else:
                self._plans[plan_id] = record
            return record, pdf_bytes

    async def readiness(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "persistent": self.persistent,
            "ready": True,
            "request_limit": self._request_limit,
        }
