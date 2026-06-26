import hashlib
import json
from typing import Any

from app.schemas.agent import AgentRunRequest


def canonical_agent_request_payload(
    request: AgentRunRequest,
    *,
    tenant_id: str,
    turn_id: str,
) -> dict[str, Any]:
    """Build the stable business payload used for Python-side turn hashing."""

    options = request.options or {}
    trusted_claims = options.get("trusted_claims") or options.get("claims") or {}
    memory_options = options.get("memory") or {}

    return {
        "tenant_id": tenant_id,
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "thread_epoch": request.thread_epoch,
        "conversation_id": request.conversation_id,
        "turn_id": turn_id,
        "user_message_id": request.user_message_id,
        "assistant_message_id": request.assistant_message_id,
        "report_id": request.report_id,
        "active_report_id": request.client_context.active_report_id,
        "message": request.message.model_dump(mode="json"),
        "memory": memory_options if isinstance(memory_options, dict) else {},
        "trusted_claims": trusted_claims if isinstance(trusted_claims, dict) else {},
    }


def canonical_agent_request_hash(
    request: AgentRunRequest,
    *,
    tenant_id: str,
    turn_id: str,
) -> str:
    payload = canonical_agent_request_payload(
        request,
        tenant_id=tenant_id,
        turn_id=turn_id,
    )
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def response_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
