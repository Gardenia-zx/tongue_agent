import base64
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import get_settings


class TurnStatus(StrEnum):
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    COMPLETED_PENDING_ACK = "COMPLETED_PENDING_ACK"
    ACKED = "ACKED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FINAL = "FAILED_FINAL"


class InboxStatus(StrEnum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    SKIPPED_PRIVACY = "SKIPPED_PRIVACY"
    FAILED = "FAILED"


class JavaAckStatus(StrEnum):
    PENDING = "PENDING"
    ACKED = "ACKED"


@dataclass(frozen=True)
class AgentTurnRecord:
    tenant_id: str
    turn_id: str
    canonical_request_hash: str
    thread_id: str
    thread_epoch: int
    status: str
    owner_id: str | None
    started_at: datetime | None
    lease_until: datetime | None
    completed_at: datetime | None
    response_snapshot_encrypted: str | None
    response_ref: dict[str, Any] | None
    response_hash: str | None
    java_ack_status: str
    failure_reason: str | None
    retry_count: int

    @classmethod
    def from_mapping(cls, row: Any) -> "AgentTurnRecord":
        data = dict(row)
        response_ref = data.get("response_ref")
        if isinstance(response_ref, str):
            try:
                response_ref = json.loads(response_ref)
            except json.JSONDecodeError:
                response_ref = None
        return cls(
            tenant_id=str(data["tenant_id"]),
            turn_id=str(data["turn_id"]),
            canonical_request_hash=str(data["canonical_request_hash"]),
            thread_id=str(data["thread_id"]),
            thread_epoch=int(data["thread_epoch"]),
            status=str(data["status"]),
            owner_id=data.get("owner_id"),
            started_at=data.get("started_at"),
            lease_until=data.get("lease_until"),
            completed_at=data.get("completed_at"),
            response_snapshot_encrypted=data.get("response_snapshot_encrypted"),
            response_ref=response_ref if isinstance(response_ref, dict) else None,
            response_hash=data.get("response_hash"),
            java_ack_status=str(data.get("java_ack_status") or JavaAckStatus.PENDING),
            failure_reason=data.get("failure_reason"),
            retry_count=int(data.get("retry_count") or 0),
        )

    @property
    def is_completed(self) -> bool:
        return self.status in {
            TurnStatus.COMPLETED_PENDING_ACK,
            TurnStatus.ACKED,
        }

    def lease_active(self, now: datetime | None = None) -> bool:
        if self.status != TurnStatus.PROCESSING or self.lease_until is None:
            return False
        return self.lease_until > (now or datetime.now(UTC))


class SnapshotCodec:
    """Authenticated encryption for pending Java ACK response snapshots.

    This deliberately uses only stdlib primitives because the project does not
    currently depend on a crypto package. Deployments should configure a stable,
    high-entropy AGENT_TURN_RECORD_ENCRYPTION_KEY.
    """

    VERSION = "hmac-xor-v1"

    def __init__(self, secret: str) -> None:
        self.key = sha256(secret.encode("utf-8")).digest()

    def encrypt(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = os.urandom(16)
        cipher = self._xor(raw, nonce)
        tag = hmac.new(
            self.key,
            self.VERSION.encode("utf-8") + nonce + cipher,
            sha256,
        ).digest()
        packed = {
            "v": self.VERSION,
            "n": self._b64(nonce),
            "c": self._b64(cipher),
            "t": self._b64(tag),
        }
        return json.dumps(packed, sort_keys=True, separators=(",", ":"))

    def decrypt(self, value: str) -> dict[str, Any]:
        packed = json.loads(value)
        if packed.get("v") != self.VERSION:
            raise ValueError("Unsupported snapshot encryption version")
        nonce = self._unb64(str(packed["n"]))
        cipher = self._unb64(str(packed["c"]))
        tag = self._unb64(str(packed["t"]))
        expected = hmac.new(
            self.key,
            self.VERSION.encode("utf-8") + nonce + cipher,
            sha256,
        ).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("Response snapshot authentication failed")
        raw = self._xor(cipher, nonce)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Response snapshot is not an object")
        return payload

    def _xor(self, data: bytes, nonce: bytes) -> bytes:
        output = bytearray()
        counter = 0
        while len(output) < len(data):
            block = hmac.new(
                self.key,
                nonce + counter.to_bytes(8, "big"),
                sha256,
            ).digest()
            output.extend(block)
            counter += 1
        return bytes(byte ^ mask for byte, mask in zip(data, output))

    @staticmethod
    def _b64(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii")

    @staticmethod
    def _unb64(value: str) -> bytes:
        return base64.urlsafe_b64decode(value.encode("ascii"))


class AgentTurnRecordStore:
    def __init__(
        self,
        *,
        engine: AsyncEngine,
        encryption_secret: str,
    ) -> None:
        self.engine = engine
        self.session_factory = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
        )
        self.codec = SnapshotCodec(encryption_secret)

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agent_turn_record (
                        tenant_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        canonical_request_hash TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        thread_epoch INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        owner_id TEXT NULL,
                        started_at TIMESTAMPTZ NULL,
                        lease_until TIMESTAMPTZ NULL,
                        completed_at TIMESTAMPTZ NULL,
                        response_snapshot_encrypted TEXT NULL,
                        response_ref JSONB NULL,
                        response_hash TEXT NULL,
                        java_ack_status TEXT NOT NULL DEFAULT 'PENDING',
                        failure_reason TEXT NULL,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (tenant_id, turn_id)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_turn_thread_status
                    ON agent_turn_record (tenant_id, thread_id, thread_epoch, status)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agent_thread_epoch (
                        tenant_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        current_epoch INTEGER NOT NULL,
                        reset_reason TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (tenant_id, thread_id)
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS memory_inbox (
                        event_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_ref JSONB NULL,
                        failure_reason TEXT NULL,
                        received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            )

    async def get(self, *, tenant_id: str, turn_id: str) -> AgentTurnRecord | None:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT * FROM agent_turn_record
                    WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                    """
                ),
                {"tenant_id": tenant_id, "turn_id": turn_id},
            )
            row = result.mappings().first()
            return AgentTurnRecord.from_mapping(row) if row else None

    async def ensure_created(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        canonical_request_hash: str,
        thread_id: str,
        thread_epoch: int,
    ) -> AgentTurnRecord:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        INSERT INTO agent_turn_record (
                            tenant_id,
                            turn_id,
                            canonical_request_hash,
                            thread_id,
                            thread_epoch,
                            status,
                            java_ack_status
                        )
                        VALUES (
                            :tenant_id,
                            :turn_id,
                            :canonical_request_hash,
                            :thread_id,
                            :thread_epoch,
                            :status,
                            :java_ack_status
                        )
                        ON CONFLICT (tenant_id, turn_id) DO NOTHING
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "turn_id": turn_id,
                        "canonical_request_hash": canonical_request_hash,
                        "thread_id": thread_id,
                        "thread_epoch": thread_epoch,
                        "status": TurnStatus.CREATED,
                        "java_ack_status": JavaAckStatus.PENDING,
                    },
                )
            result = await session.execute(
                text(
                    """
                    SELECT * FROM agent_turn_record
                    WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                    """
                ),
                {"tenant_id": tenant_id, "turn_id": turn_id},
            )
            row = result.mappings().first()
            if row is None:
                raise RuntimeError("agent_turn_record insert/read failed")
            return AgentTurnRecord.from_mapping(row)

    async def mark_processing(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        owner_id: str,
        lease_seconds: int,
    ) -> AgentTurnRecord:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=lease_seconds)
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        """
                        UPDATE agent_turn_record
                        SET status = :status,
                            owner_id = :owner_id,
                            started_at = COALESCE(started_at, :started_at),
                            lease_until = :lease_until,
                            retry_count = CASE
                                WHEN status = :failed_retryable THEN retry_count + 1
                                ELSE retry_count
                            END,
                            failure_reason = NULL,
                            updated_at = NOW()
                        WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                        RETURNING *
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "turn_id": turn_id,
                        "status": TurnStatus.PROCESSING,
                        "owner_id": owner_id,
                        "started_at": now,
                        "lease_until": lease_until,
                        "failed_retryable": TurnStatus.FAILED_RETRYABLE,
                    },
                )
                row = result.mappings().first()
            if row is None:
                raise RuntimeError("agent_turn_record mark_processing failed")
            return AgentTurnRecord.from_mapping(row)

    async def mark_completed_pending_ack(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        response_snapshot: dict[str, Any],
        response_hash: str,
    ) -> AgentTurnRecord:
        encrypted = self.codec.encrypt(response_snapshot)
        now = datetime.now(UTC)
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        """
                        UPDATE agent_turn_record
                        SET status = :status,
                            completed_at = :completed_at,
                            response_snapshot_encrypted = :response_snapshot_encrypted,
                            response_hash = :response_hash,
                            java_ack_status = :java_ack_status,
                            lease_until = NULL,
                            updated_at = NOW()
                        WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                        RETURNING *
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "turn_id": turn_id,
                        "status": TurnStatus.COMPLETED_PENDING_ACK,
                        "completed_at": now,
                        "response_snapshot_encrypted": encrypted,
                        "response_hash": response_hash,
                        "java_ack_status": JavaAckStatus.PENDING,
                    },
                )
                row = result.mappings().first()
            if row is None:
                raise RuntimeError("agent_turn_record mark_completed failed")
            return AgentTurnRecord.from_mapping(row)

    async def mark_failed(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        retryable: bool,
        reason: str,
    ) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        UPDATE agent_turn_record
                        SET status = :status,
                            failure_reason = :failure_reason,
                            lease_until = NULL,
                            updated_at = NOW()
                        WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "turn_id": turn_id,
                        "status": (
                            TurnStatus.FAILED_RETRYABLE
                            if retryable
                            else TurnStatus.FAILED_FINAL
                        ),
                        "failure_reason": reason[:1000],
                    },
                )

    async def ack(
        self,
        *,
        tenant_id: str,
        turn_id: str,
        assistant_message_id: str,
        response_hash: str,
    ) -> AgentTurnRecord | None:
        record = await self.get(tenant_id=tenant_id, turn_id=turn_id)
        if record is None:
            return None

        if record.response_hash != response_hash:
            raise ValueError("response_hash_mismatch")

        existing_ref = record.response_ref or {}
        if record.status == TurnStatus.ACKED:
            if existing_ref.get("assistant_message_id") != assistant_message_id:
                raise ValueError("assistant_message_id_mismatch")
            return record

        if record.status not in {
            TurnStatus.COMPLETED_PENDING_ACK,
            TurnStatus.ACKED,
        }:
            raise ValueError("turn_not_completed")

        response_ref = {
            "assistant_message_id": assistant_message_id,
            "response_hash": response_hash,
        }
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        """
                        UPDATE agent_turn_record
                        SET status = :status,
                            response_snapshot_encrypted = NULL,
                            response_ref = CAST(:response_ref AS JSONB),
                            java_ack_status = :java_ack_status,
                            updated_at = NOW()
                        WHERE tenant_id = :tenant_id AND turn_id = :turn_id
                        RETURNING *
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "turn_id": turn_id,
                        "status": TurnStatus.ACKED,
                        "response_ref": json.dumps(response_ref, ensure_ascii=False),
                        "java_ack_status": JavaAckStatus.ACKED,
                    },
                )
                row = result.mappings().first()
        return AgentTurnRecord.from_mapping(row) if row else None

    async def list_pending_ack(
        self,
        *,
        older_than_seconds: int,
        limit: int = 100,
    ) -> list[AgentTurnRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT *
                    FROM agent_turn_record
                    WHERE status = 'COMPLETED_PENDING_ACK'
                      AND completed_at IS NOT NULL
                      AND completed_at < NOW() - (:older_than_seconds * INTERVAL '1 second')
                    ORDER BY completed_at ASC
                    LIMIT :limit
                    """
                ),
                {
                    "older_than_seconds": older_than_seconds,
                    "limit": limit,
                },
            )
            return [
                AgentTurnRecord.from_mapping(row)
                for row in result.mappings().all()
            ]

    def decrypt_response_snapshot(self, record: AgentTurnRecord) -> dict[str, Any] | None:
        if not record.response_snapshot_encrypted:
            return None
        return self.codec.decrypt(record.response_snapshot_encrypted)

    async def get_thread_epoch(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> int | None:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT current_epoch FROM agent_thread_epoch
                    WHERE tenant_id = :tenant_id AND thread_id = :thread_id
                    """
                ),
                {"tenant_id": tenant_id, "thread_id": thread_id},
            )
            value = result.scalar_one_or_none()
            return int(value) if value is not None else None

    async def ensure_thread_epoch(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        thread_epoch: int,
    ) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        INSERT INTO agent_thread_epoch (
                            tenant_id,
                            thread_id,
                            current_epoch
                        )
                        VALUES (:tenant_id, :thread_id, :thread_epoch)
                        ON CONFLICT (tenant_id, thread_id) DO NOTHING
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "thread_id": thread_id,
                        "thread_epoch": thread_epoch,
                    },
                )
            current = await self.get_thread_epoch(
                tenant_id=tenant_id,
                thread_id=thread_id,
            )
            if current is None:
                raise RuntimeError("agent_thread_epoch insert/read failed")
            return current

    async def active_turn_count(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        exclude_turn_id: str | None = None,
    ) -> int:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM agent_turn_record
                    WHERE tenant_id = :tenant_id
                      AND thread_id = :thread_id
                      AND status IN ('CREATED', 'PROCESSING')
                      AND (lease_until IS NULL OR lease_until > NOW())
                      AND (:exclude_turn_id IS NULL OR turn_id <> :exclude_turn_id)
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "thread_id": thread_id,
                    "exclude_turn_id": exclude_turn_id,
                },
            )
            return int(result.scalar_one())

    async def advance_thread_epoch(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        new_epoch: int,
        reset_reason: str,
    ) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        INSERT INTO agent_thread_epoch (
                            tenant_id,
                            thread_id,
                            current_epoch,
                            reset_reason
                        )
                        VALUES (:tenant_id, :thread_id, :new_epoch, :reset_reason)
                        ON CONFLICT (tenant_id, thread_id)
                        DO UPDATE SET
                            current_epoch = :new_epoch,
                            reset_reason = :reset_reason,
                            updated_at = NOW()
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "thread_id": thread_id,
                        "new_epoch": new_epoch,
                        "reset_reason": reset_reason,
                    },
                )


def create_turn_record_engine() -> AsyncEngine:
    return create_async_engine(
        get_settings().rag_database_url,
        pool_pre_ping=True,
    )


def create_turn_record_store(engine: AsyncEngine) -> AgentTurnRecordStore:
    settings = get_settings()
    secret = (
        settings.agent_turn_record_encryption_key
        or f"{settings.app_name}:{settings.app_env}:{settings.langgraph_postgres_uri}"
    )
    return AgentTurnRecordStore(
        engine=engine,
        encryption_secret=secret,
    )
