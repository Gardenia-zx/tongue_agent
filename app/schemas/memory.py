from typing import Any, Literal

from pydantic import BaseModel, Field


class LongTermMemoryRecord(BaseModel):
    schema_version: str = "1.0"
    memory_id: str
    memory_type: str
    memory_key: str
    text: str
    summary: str
    value: str
    status: Literal["ACTIVE", "SUPERSEDED", "ARCHIVED"] = "ACTIVE"
    source: str = "explicit_user_input"
    created_at: str
    updated_at: str
    supersedes: str | None = None


class UserProfileMemory(BaseModel):
    schema_version: str = "1.0"
    profile_id: str
    preferences: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    created_at: str
    updated_at: str


class MemorySummaryRecord(BaseModel):
    schema_version: str = "1.0"
    summary_id: str
    memory_type: str
    text: str
    summary: str
    status: Literal["ACTIVE", "SUPERSEDED", "ARCHIVED"] = "ACTIVE"
    source_memory_count: int = 0
    updated_at: str


class MemoryAuditRecord(BaseModel):
    schema_version: str = "1.0"
    audit_id: str
    action: str
    memory_key: str
    request_id: str | None = None
    old_memory_id: str | None = None
    new_memory_id: str | None = None
    created_at: str
