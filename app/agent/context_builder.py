import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.state import AgentState


ContextMode = Literal["MINIMAL_PRE_INTENT", "FULL_FOR_NODE"]
ShortTermSource = Literal["CHECKPOINTER", "MYSQL_RECOVERY", "LEGACY_BUNDLE"]

CHECKPOINTER_SOURCE = "CHECKPOINTER"
MYSQL_RECOVERY_SOURCE = "MYSQL_RECOVERY"
LEGACY_BUNDLE_SOURCE = "LEGACY_BUNDLE"

DEFAULT_TOKEN_BUDGET = 3600
MIN_TOKEN_BUDGET = 600


class ContextSourceRef(BaseModel):
    source: str
    source_id: Any = None
    created_at: str | None = None
    node_name: str | None = None
    answer_type: str | None = None


class ContextMessage(BaseModel):
    tenant_id: str | None = None
    conversation_id: str | None = None
    message_id: Any = None
    external_message_id: Any = None
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    content_type: str = "text"
    report_id: Any = None
    source: str | None = None
    source_id: Any = None
    created_at: str | None = None
    node_name: str | None = None
    answer_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured_content: dict[str, Any] = Field(default_factory=dict)
    sources: list[ContextSourceRef] = Field(default_factory=list)


class ShortTermContext(BaseModel):
    source: ShortTermSource
    fallback_reason: str | None = None
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    recent_messages: list[ContextMessage] = Field(default_factory=list)
    last_final_answer: dict[str, Any] | None = None
    conversation_summary: dict[str, Any] | str | None = None
    active_report_ref: dict[str, Any] | None = None
    integrity_status: str = "OK"


class ContextHealth(BaseModel):
    status: Literal["OK", "PARTIAL", "EMPTY"] = "EMPTY"
    source: str | None = None
    integrity_status: str = "OK"
    has_recent_messages: bool = False
    has_recent_complete_turn: bool = False
    has_last_final_answer: bool = False
    has_conversation_summary: bool = False
    has_active_report_ref: bool = False
    missing: list[str] = Field(default_factory=list)


class QueryRewriteContext(BaseModel):
    schema_version: str = "1.0"
    turn_id: str | None = None
    context_version: str = "query_rewrite_context.v1"
    raw_user_input: str = ""
    recent_messages: list[ContextMessage] = Field(default_factory=list)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    last_final_answer: dict[str, Any] | None = None
    conversation_summary: dict[str, Any] | str | None = None
    active_report_ref: dict[str, Any] | None = None
    context_health: ContextHealth = Field(default_factory=ContextHealth)
    token_budget: int = DEFAULT_TOKEN_BUDGET
    context_sources: list[dict[str, Any]] = Field(default_factory=list)


class ReferenceResolution(BaseModel):
    status: str = "NOT_NEEDED"
    confidence: float = 0.0
    target_type: str = "UNKNOWN"
    target_focus: str = "UNKNOWN"
    target_message_id: Any = None
    target_report_id: Any = None
    evidence_sources: list[dict[str, Any]] = Field(default_factory=list)
    is_context_dependent: bool = False
    reason: str = ""


class QueryContext(BaseModel):
    schema_version: str = "1.0"
    turn_id: str | None = None
    raw_user_input: str = ""
    standalone_query: str = ""
    reference_resolution: ReferenceResolution = Field(default_factory=ReferenceResolution)
    clarification_status: str = "NOT_NEEDED"
    route_hint: str | None = None
    rewrite_confidence: float = 0.0
    prompt_context: dict[str, Any] = Field(default_factory=dict)


class FinalPromptContext(BaseModel):
    schema_version: str = "1.0"
    mode: ContextMode = "FULL_FOR_NODE"
    node_name: str | None = None
    system_prompt: str = ""
    raw_user_input: str = ""
    standalone_query: str = ""
    reference_resolution: dict[str, Any] = Field(default_factory=dict)
    reference_target: dict[str, Any] | None = None
    intent_result: dict[str, Any] = Field(default_factory=dict)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    conversation_summary: dict[str, Any] | str | None = None
    last_final_answer: dict[str, Any] | None = None
    active_report: dict[str, Any] | None = None
    active_report_ref: dict[str, Any] | None = None
    business_context: dict[str, Any] = Field(default_factory=dict)
    privacy_policy: dict[str, Any] = Field(default_factory=dict)
    related_long_term_memories: list[dict[str, Any]] = Field(default_factory=list)
    context_health: dict[str, Any] = Field(default_factory=dict)
    context_sources: list[dict[str, Any]] = Field(default_factory=list)
    token_budget: int = DEFAULT_TOKEN_BUDGET
    trim_trace: list[str] = Field(default_factory=list)


class CurrentTurnState(BaseModel):
    schema_version: str = "1.0"
    turn_id: str
    short_term_context: dict[str, Any] = Field(default_factory=dict)
    query_rewrite_context: dict[str, Any] = Field(default_factory=dict)
    query_context: dict[str, Any] = Field(default_factory=dict)
    business_context: dict[str, Any] = Field(default_factory=dict)
    final_prompt_context: dict[str, Any] = Field(default_factory=dict)
    context_health: dict[str, Any] = Field(default_factory=dict)


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def current_turn_id_from_state(state: AgentState) -> str:
    value = state.get("turn_id") or state.get("request_id") or ""
    return str(value)


def new_current_turn(turn_id: str) -> dict[str, Any]:
    return _dump(CurrentTurnState(turn_id=turn_id))


def ensure_current_turn(state: AgentState) -> dict[str, Any]:
    turn_id = current_turn_id_from_state(state)
    current_turn = state.get("current_turn") or {}
    if isinstance(current_turn, dict) and current_turn.get("turn_id") == turn_id:
        return deepcopy(current_turn)
    return new_current_turn(turn_id)


def with_current_turn(state: AgentState, current_turn: dict[str, Any]) -> AgentState:
    return {
        **state,
        "current_turn": current_turn,
    }


def current_turn_from_state(state: AgentState) -> dict[str, Any]:
    current_turn = state.get("current_turn") or {}
    return current_turn if isinstance(current_turn, dict) else {}


def context_bundle_from_state(state: AgentState) -> dict[str, Any]:
    context_bundle = state.get("context_bundle") or {}
    if isinstance(context_bundle, dict) and context_bundle:
        return context_bundle

    client_context = state.get("client_context") or {}
    client_extra = client_context.get("extra") or {}
    extra_bundle = client_extra.get("context_bundle") or {}
    return extra_bundle if isinstance(extra_bundle, dict) else {}


def _client_extra(state: AgentState) -> dict[str, Any]:
    client_context = state.get("client_context") or {}
    if not isinstance(client_context, dict):
        return {}
    extra = client_context.get("extra") or {}
    return extra if isinstance(extra, dict) else {}


def _scope(state: AgentState) -> tuple[str, str]:
    tenant_id = str(state.get("tenant_id") or state.get("user_id") or "")
    bundle = context_bundle_from_state(state)
    conversation_id = str(
        state.get("conversation_id")
        or bundle.get("conversation_id")
        or state.get("thread_id")
        or ""
    )
    return tenant_id, conversation_id


def _metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    metadata_json = message.get("metadata_json")
    return metadata_json if isinstance(metadata_json, dict) else {}


def _normalized_hash(content: str) -> str:
    normalized = " ".join(content.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _message_key(
    *,
    state: AgentState,
    message: dict[str, Any],
) -> str:
    tenant_id, conversation_id = _scope(state)
    role = str(message.get("role") or "")
    stable_id = message.get("message_id") or message.get("external_message_id")
    if stable_id:
        return f"id:{tenant_id}:{conversation_id}:{stable_id}:{role}"
    return f"hash:{tenant_id}:{conversation_id}:{role}:{_normalized_hash(str(message.get('content') or ''))}"


def _source_ref(
    *,
    primary_source: str,
    message: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": str(message.get("source") or primary_source),
        "source_id": message.get("source_id")
        or message.get("message_id")
        or message.get("external_message_id"),
        "created_at": message.get("created_at"),
        "node_name": message.get("node_name"),
        "answer_type": message.get("answer_type"),
    }


def _merge_source_refs(
    existing: list[dict[str, Any]],
    new_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in [*existing, *new_refs]:
        if not isinstance(ref, dict):
            continue
        key = json.dumps(ref, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def _normalize_message(
    state: AgentState,
    item: dict[str, Any],
    *,
    primary_source: str,
) -> dict[str, Any] | None:
    role = item.get("role")
    if role not in {"user", "assistant", "system", "tool"}:
        return None
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    metadata = _metadata(item)
    structured_content = (
        metadata.get("structured_content")
        or item.get("structured_content")
        or {}
    )
    if not isinstance(structured_content, dict):
        structured_content = {}

    tenant_id, conversation_id = _scope(state)
    node_name = item.get("node_name") or metadata.get("node_name")
    answer_type = item.get("answer_type") or metadata.get("answer_type")
    normalized = {
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "message_id": item.get("message_id"),
        "external_message_id": item.get("external_message_id"),
        "role": role,
        "content": content.strip(),
        "content_type": item.get("content_type") or "text",
        "report_id": item.get("report_id") or metadata.get("report_id"),
        "source": item.get("source") or primary_source,
        "source_id": item.get("source_id"),
        "created_at": item.get("created_at"),
        "node_name": node_name,
        "answer_type": answer_type,
        "metadata": metadata,
        "structured_content": structured_content,
    }
    raw_sources = item.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    normalized["sources"] = _merge_source_refs(
        sources,
        [_source_ref(primary_source=primary_source, message=normalized)],
    )
    return _dump(ContextMessage(**normalized))


def _messages_from_turns(
    state: AgentState,
    turns: list[Any],
    *,
    primary_source: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") and turn.get("content"):
            message = _normalize_message(state, turn, primary_source=primary_source)
            if message:
                messages.append(message)
            continue

        created_at = turn.get("created_at")
        source_id = turn.get("turn_id") or turn.get("request_id")
        user_text = turn.get("user")
        if isinstance(user_text, str) and user_text.strip():
            message = _normalize_message(
                state,
                {
                    "role": "user",
                    "content": user_text,
                    "created_at": created_at,
                    "source_id": source_id,
                },
                primary_source=primary_source,
            )
            if message:
                messages.append(message)

        assistant_text = turn.get("assistant")
        if isinstance(assistant_text, str) and assistant_text.strip():
            message = _normalize_message(
                state,
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "created_at": created_at,
                    "source_id": source_id,
                    "node_name": turn.get("node"),
                    "answer_type": turn.get("answer_type"),
                    "report_id": turn.get("report_id"),
                },
                primary_source=primary_source,
            )
            if message:
                messages.append(message)
    return messages


def _dedupe_messages(
    state: AgentState,
    messages: list[dict[str, Any]],
    *,
    primary_source: str,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized = _normalize_message(state, message, primary_source=primary_source)
        if not normalized:
            continue
        key = _message_key(state=state, message=normalized)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = normalized
            continue

        merged = dict(existing)
        for field in (
            "message_id",
            "external_message_id",
            "content_type",
            "report_id",
            "source_id",
            "created_at",
            "node_name",
            "answer_type",
        ):
            if not merged.get(field) and normalized.get(field):
                merged[field] = normalized[field]
        merged["metadata"] = {
            **(existing.get("metadata") or {}),
            **(normalized.get("metadata") or {}),
        }
        merged["structured_content"] = (
            existing.get("structured_content") or normalized.get("structured_content") or {}
        )
        merged["sources"] = _merge_source_refs(
            existing.get("sources") or [],
            normalized.get("sources") or [],
        )
        by_key[key] = merged
    return list(by_key.values())[-12:]


def _assistant_answer_context(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    metadata = _metadata(message)
    structured_content = (
        metadata.get("structured_content")
        or message.get("structured_content")
        or {}
    )
    if not isinstance(structured_content, dict):
        structured_content = {}

    return {
        "message_id": message.get("message_id"),
        "external_message_id": message.get("external_message_id"),
        "role": "assistant",
        "content": content.strip(),
        "content_type": message.get("content_type") or "text",
        "report_id": message.get("report_id") or metadata.get("report_id"),
        "node_name": message.get("node_name") or metadata.get("node_name"),
        "route_target": message.get("route_target") or metadata.get("route_target"),
        "answer_type": message.get("answer_type") or metadata.get("answer_type"),
        "rag_query": message.get("rag_query") or metadata.get("rag_query"),
        "structured_content": structured_content,
        "metadata": metadata,
        "created_at": message.get("created_at"),
        "sources": message.get("sources") or [],
    }


def _last_assistant_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        answer_context = _assistant_answer_context(message)
        if answer_context:
            return answer_context
    return None


def _summary_text(summary: Any) -> str:
    if isinstance(summary, str):
        return summary.strip()
    if isinstance(summary, dict):
        return str(summary.get("text") or summary.get("summary") or "").strip()
    return ""


def _active_report_candidates(state: AgentState) -> list[dict[str, Any]]:
    bundle = context_bundle_from_state(state)
    client_extra = _client_extra(state)
    candidates = [
        bundle.get("active_report"),
        client_extra.get("active_report"),
        client_extra.get("latest_report"),
        client_extra.get("latest_report_context"),
        client_extra.get("frontend_latest_report"),
    ]
    return [item for item in candidates if isinstance(item, dict) and item]


def _report_ref(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict) or not report:
        return None
    ref: dict[str, Any] = {}
    for key in (
        "report_id",
        "id",
        "task_id",
        "task_version",
        "owner_user_id",
        "tenant_id",
        "created_at",
        "updated_at",
        "feature_summary",
        "featureSummary",
        "summary",
    ):
        if report.get(key) is not None:
            ref[key] = report.get(key)
    if "report_id" not in ref and ref.get("id") is not None:
        ref["report_id"] = ref.get("id")
    return ref or None


def _active_report_ref_from_state(state: AgentState) -> dict[str, Any] | None:
    for report in _active_report_candidates(state):
        ref = _report_ref(report)
        if ref:
            return ref
    return None


def _active_report_full_from_state(state: AgentState) -> dict[str, Any] | None:
    current_turn = current_turn_from_state(state)
    final_prompt = current_turn.get("final_prompt_context") or {}
    if isinstance(final_prompt, dict):
        active_report = final_prompt.get("active_report")
        if isinstance(active_report, dict) and active_report:
            return active_report
    business_context = current_turn.get("business_context") or {}
    if isinstance(business_context, dict):
        active_report = business_context.get("active_report")
        if isinstance(active_report, dict) and active_report:
            return active_report
    for report in _active_report_candidates(state):
        return report
    return None


def _checkpoint_payload(state: AgentState) -> dict[str, Any] | None:
    memory_context = state.get("memory_context") or {}
    if not isinstance(memory_context, dict):
        return None
    session = memory_context.get("session") or {}
    if isinstance(session, dict) and session.get("source") == "backend_context_bundle":
        return None

    recent_turns = memory_context.get("recent_turns")
    summary = memory_context.get("conversation_summary")
    has_material = bool(recent_turns) or bool(_summary_text(summary))
    if not has_material and not (isinstance(session, dict) and session.get("cache_hit")):
        return None
    return memory_context


def _mysql_recovery_payload(state: AgentState) -> dict[str, Any] | None:
    candidates = [
        state.get("mysql_recovery_context"),
        state.get("recovery_context"),
        _client_extra(state).get("mysql_recovery_context"),
        _client_extra(state).get("recovery_context"),
    ]
    bundle = context_bundle_from_state(state)
    if bundle.get("source") == MYSQL_RECOVERY_SOURCE or bundle.get("mode") == "mysql_recovery":
        candidates.append(bundle)
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def _legacy_bundle_payload(state: AgentState) -> dict[str, Any] | None:
    bundle = context_bundle_from_state(state)
    if isinstance(bundle, dict) and bundle:
        return bundle
    client_extra = _client_extra(state)
    legacy = {
        "recent_messages": client_extra.get("recent_messages") or [],
        "last_final_answer": client_extra.get("last_final_answer") or {},
        "active_report": (
            client_extra.get("active_report")
            or client_extra.get("latest_report")
            or client_extra.get("latest_report_context")
            or client_extra.get("frontend_latest_report")
        ),
        "conversation_summary": client_extra.get("conversation_summary"),
    }
    if legacy["recent_messages"] or legacy["last_final_answer"] or legacy["active_report"]:
        return legacy
    return None


def _payload_recent_messages(
    state: AgentState,
    payload: dict[str, Any],
    *,
    primary_source: str,
) -> list[dict[str, Any]]:
    recent_messages = payload.get("recent_messages")
    if not isinstance(recent_messages, list):
        recent_messages = []

    recent_turns = payload.get("recent_turns")
    if not isinstance(recent_turns, list):
        recent_turns = []

    messages = [
        *recent_messages,
        *_messages_from_turns(state, recent_turns, primary_source=primary_source),
    ]
    return _dedupe_messages(state, messages, primary_source=primary_source)


def _payload_recent_turns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recent_turns = payload.get("recent_turns")
    if isinstance(recent_turns, list):
        return [dict(item) for item in recent_turns if isinstance(item, dict)][-8:]
    return []


def _payload_last_answer(
    state: AgentState,
    payload: dict[str, Any],
    *,
    primary_source: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    bundled = payload.get("last_final_answer")
    if isinstance(bundled, dict) and bundled.get("content"):
        normalized = _normalize_message(state, bundled, primary_source=primary_source)
        if normalized:
            return _assistant_answer_context(normalized)
        return _assistant_answer_context(bundled)
    return _last_assistant_from_messages(messages)


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any] | str | None:
    summary = payload.get("conversation_summary")
    if summary:
        return summary
    summary = payload.get("summary")
    return summary if summary else None


def _payload_report_ref(state: AgentState, payload: dict[str, Any]) -> dict[str, Any] | None:
    report = payload.get("active_report") or payload.get("latest_report")
    if isinstance(report, dict) and report:
        return _report_ref(report)
    return _active_report_ref_from_state(state)


def _integrity_status(payload: dict[str, Any]) -> str:
    explicit = payload.get("integrity_status")
    if isinstance(explicit, str) and explicit:
        return explicit
    if payload.get("skip_reason"):
        return "DEGRADED"
    return "OK"


def select_short_term_context(state: AgentState) -> dict[str, Any]:
    fallback_reason: str | None = None
    checkpoint_payload = _checkpoint_payload(state)
    if checkpoint_payload is not None:
        return _build_short_term_context(
            state,
            payload=checkpoint_payload,
            source=CHECKPOINTER_SOURCE,
            fallback_reason=None,
        )

    fallback_reason = "checkpointer_context_missing"
    mysql_payload = _mysql_recovery_payload(state)
    if mysql_payload is not None:
        return _build_short_term_context(
            state,
            payload=mysql_payload,
            source=MYSQL_RECOVERY_SOURCE,
            fallback_reason=fallback_reason,
        )

    fallback_reason = "mysql_recovery_context_missing"
    legacy_payload = _legacy_bundle_payload(state)
    if legacy_payload is not None:
        return _build_short_term_context(
            state,
            payload=legacy_payload,
            source=LEGACY_BUNDLE_SOURCE,
            fallback_reason=fallback_reason,
        )

    return _dump(
        ShortTermContext(
            source=LEGACY_BUNDLE_SOURCE,
            fallback_reason="all_short_term_sources_missing",
            integrity_status="EMPTY",
        )
    )


def _build_short_term_context(
    state: AgentState,
    *,
    payload: dict[str, Any],
    source: ShortTermSource,
    fallback_reason: str | None,
) -> dict[str, Any]:
    messages = _payload_recent_messages(state, payload, primary_source=source)
    recent_turns = _payload_recent_turns(payload)
    return _dump(
        ShortTermContext(
            source=source,
            fallback_reason=fallback_reason,
            recent_turns=recent_turns,
            recent_messages=[ContextMessage(**message) for message in messages],
            last_final_answer=_payload_last_answer(
                state,
                payload,
                primary_source=source,
                messages=messages,
            ),
            conversation_summary=_payload_summary(payload),
            active_report_ref=_payload_report_ref(state, payload),
            integrity_status=_integrity_status(payload),
        )
    )


def _has_complete_turn(short_term_context: dict[str, Any]) -> bool:
    turns = short_term_context.get("recent_turns") or []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("user") or "").strip() and str(turn.get("assistant") or "").strip():
            return True
    messages = short_term_context.get("recent_messages") or []
    has_user = any(isinstance(item, dict) and item.get("role") == "user" for item in messages)
    has_assistant = any(
        isinstance(item, dict) and item.get("role") == "assistant" for item in messages
    )
    return has_user and has_assistant


def build_context_health(short_term_context: dict[str, Any]) -> dict[str, Any]:
    recent_messages = short_term_context.get("recent_messages") or []
    last_answer = short_term_context.get("last_final_answer")
    summary = short_term_context.get("conversation_summary")
    active_report_ref = short_term_context.get("active_report_ref")
    has_recent = bool(recent_messages)
    has_last = isinstance(last_answer, dict) and bool(last_answer.get("content"))
    has_summary = bool(_summary_text(summary))
    has_report = isinstance(active_report_ref, dict) and bool(active_report_ref)
    missing = []
    if not has_recent:
        missing.append("recent_messages")
    if not has_last:
        missing.append("last_final_answer")
    if not has_summary:
        missing.append("conversation_summary")
    if not has_report:
        missing.append("active_report_ref")

    material_count = sum([has_recent, has_last, has_summary, has_report])
    status_value: Literal["OK", "PARTIAL", "EMPTY"]
    if material_count >= 2:
        status_value = "OK"
    elif material_count == 1:
        status_value = "PARTIAL"
    else:
        status_value = "EMPTY"

    return _dump(
        ContextHealth(
            status=status_value,
            source=short_term_context.get("source"),
            integrity_status=short_term_context.get("integrity_status") or "OK",
            has_recent_messages=has_recent,
            has_recent_complete_turn=_has_complete_turn(short_term_context),
            has_last_final_answer=has_last,
            has_conversation_summary=has_summary,
            has_active_report_ref=has_report,
            missing=missing,
        )
    )


def token_budget_from_state(state: AgentState) -> int:
    options = state.get("options") or {}
    context_options = options.get("context") if isinstance(options, dict) else {}
    candidates = []
    if isinstance(context_options, dict):
        candidates.append(context_options.get("token_budget"))
    candidates.extend([options.get("token_budget") if isinstance(options, dict) else None])
    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        return max(MIN_TOKEN_BUDGET, value)
    return DEFAULT_TOKEN_BUDGET


def build_query_rewrite_context(state: AgentState) -> dict[str, Any]:
    short_term_context = select_short_term_context(state)
    context_health = build_context_health(short_term_context)
    token_budget = token_budget_from_state(state)
    context_sources = [
        {
            "source": short_term_context.get("source"),
            "fallback_reason": short_term_context.get("fallback_reason"),
            "integrity_status": short_term_context.get("integrity_status"),
            "message_count": len(short_term_context.get("recent_messages") or []),
            "turn_count": len(short_term_context.get("recent_turns") or []),
        }
    ]
    return _dump(
        QueryRewriteContext(
            turn_id=current_turn_id_from_state(state),
            raw_user_input=extract_user_text(state),
            recent_messages=[
                ContextMessage(**item)
                for item in (short_term_context.get("recent_messages") or [])
                if isinstance(item, dict)
            ],
            recent_turns=short_term_context.get("recent_turns") or [],
            last_final_answer=short_term_context.get("last_final_answer"),
            conversation_summary=short_term_context.get("conversation_summary"),
            active_report_ref=short_term_context.get("active_report_ref"),
            context_health=ContextHealth(**context_health),
            token_budget=token_budget,
            context_sources=context_sources,
        )
    )


def with_query_rewrite_context(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    existing_context = current_turn.get("query_rewrite_context")
    if isinstance(existing_context, dict) and existing_context:
        return {
            **state,
            "current_turn": current_turn,
            "prompt_context": {
                "schema_version": "1.0",
                "mode": "MINIMAL_PRE_INTENT",
                **existing_context,
            },
        }

    query_rewrite_context = build_query_rewrite_context(state)
    short_term_context = select_short_term_context(state)
    current_turn["short_term_context"] = short_term_context
    current_turn["query_rewrite_context"] = query_rewrite_context
    current_turn["context_health"] = query_rewrite_context.get("context_health") or {}
    return {
        **state,
        "current_turn": current_turn,
        "prompt_context": {
            "schema_version": "1.0",
            "mode": "MINIMAL_PRE_INTENT",
            **query_rewrite_context,
        },
    }


def query_rewrite_context_from_state(state: AgentState) -> dict[str, Any]:
    current_turn = current_turn_from_state(state)
    value = current_turn.get("query_rewrite_context")
    if isinstance(value, dict) and value:
        return value
    return build_query_rewrite_context(state)


def query_context_from_state(state: AgentState) -> dict[str, Any]:
    current_turn = current_turn_from_state(state)
    value = current_turn.get("query_context")
    if isinstance(value, dict) and value:
        return value
    value = state.get("query_context") or {}
    return value if isinstance(value, dict) else {}


def with_query_context(state: AgentState, query_context: dict[str, Any]) -> AgentState:
    current_turn = ensure_current_turn(state)
    current_turn["query_context"] = query_context
    return {
        **state,
        "current_turn": current_turn,
        "query_context": query_context,
    }


def effective_user_query(state: AgentState) -> str:
    query_context = query_context_from_state(state)
    standalone_query = query_context.get("standalone_query")
    if isinstance(standalone_query, str) and standalone_query.strip():
        return standalone_query.strip()
    return extract_user_text(state)


def recent_messages_from_state(state: AgentState) -> list[dict[str, Any]]:
    current_turn = current_turn_from_state(state)
    for container_name in ("final_prompt_context", "query_rewrite_context", "short_term_context"):
        container = current_turn.get(container_name) or {}
        if isinstance(container, dict):
            messages = container.get("recent_messages")
            if isinstance(messages, list) and messages:
                return [item for item in messages if isinstance(item, dict)][-12:]

    bundle = context_bundle_from_state(state)
    raw_messages = bundle.get("recent_messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    return _dedupe_messages(state, raw_messages, primary_source=LEGACY_BUNDLE_SOURCE)


def active_report_from_state(state: AgentState) -> dict[str, Any] | None:
    current_turn = current_turn_from_state(state)
    for container_name in ("final_prompt_context", "business_context"):
        container = current_turn.get(container_name) or {}
        if isinstance(container, dict):
            active_report = container.get("active_report")
            if isinstance(active_report, dict) and active_report:
                return active_report
    for container_name in ("query_rewrite_context", "short_term_context"):
        container = current_turn.get(container_name) or {}
        if isinstance(container, dict):
            active_report_ref = container.get("active_report_ref")
            if isinstance(active_report_ref, dict) and active_report_ref:
                return active_report_ref
    return _active_report_full_from_state(state)


def conversation_summary_from_state(state: AgentState) -> dict[str, Any] | str | None:
    current_turn = current_turn_from_state(state)
    for container_name in ("final_prompt_context", "query_rewrite_context", "short_term_context"):
        container = current_turn.get(container_name) or {}
        if isinstance(container, dict) and container.get("conversation_summary"):
            return container.get("conversation_summary")

    bundle = context_bundle_from_state(state)
    summary = bundle.get("conversation_summary")
    if summary:
        return summary

    memory_context = state.get("memory_context") or {}
    summary_text = memory_context.get("conversation_summary")
    return summary_text if summary_text else None


def traceback_context_from_state(state: AgentState) -> dict[str, Any]:
    bundle = context_bundle_from_state(state)
    traceback_context = bundle.get("traceback_context") or {}
    return traceback_context if isinstance(traceback_context, dict) else {}


def last_final_answer_from_state(state: AgentState) -> dict[str, Any] | None:
    current_turn = current_turn_from_state(state)
    for container_name in ("query_rewrite_context", "short_term_context", "final_prompt_context"):
        container = current_turn.get(container_name) or {}
        if isinstance(container, dict):
            answer = container.get("last_final_answer")
            if isinstance(answer, dict) and answer.get("content"):
                return answer

    bundle = context_bundle_from_state(state)
    bundled = bundle.get("last_final_answer")
    if isinstance(bundled, dict) and bundled.get("content"):
        return _assistant_answer_context(bundled)

    return _last_assistant_from_messages(recent_messages_from_state(state))


def related_long_term_memories_from_state(state: AgentState) -> list[dict[str, Any]]:
    memory_context = state.get("memory_context") or {}
    memories = memory_context.get("relevant_memories") or memory_context.get("memories") or []
    summaries = memory_context.get("summaries") or []
    profile = memory_context.get("profile") or {}

    result: list[dict[str, Any]] = []
    for item in memories:
        if isinstance(item, dict):
            result.append({"source": "long_term_memory", **item})
    for item in summaries:
        if isinstance(item, dict):
            result.append({"source": "memory_summary", **item})
    if isinstance(profile, dict) and profile:
        result.append({"source": "profile", **profile})
    return result[:8]


def _business_context_from_state(state: AgentState) -> dict[str, Any]:
    current_turn = current_turn_from_state(state)
    business_context = current_turn.get("business_context")
    if isinstance(business_context, dict) and business_context:
        return business_context
    value = state.get("business_context")
    return value if isinstance(value, dict) else {}


def _privacy_policy_from_state(state: AgentState) -> dict[str, Any]:
    options = state.get("options") or {}
    if isinstance(options, dict):
        policy = options.get("privacy_policy") or options.get("privacy")
        if isinstance(policy, dict):
            return policy
    memory_context = state.get("memory_context") or {}
    policy = memory_context.get("policy")
    return policy if isinstance(policy, dict) else {}


def _estimated_chars(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _content_trimmed_messages(messages: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages[-limit:]:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str) and len(content) > 900:
            item["content"] = content[:900].rstrip()
        result.append(item)
    return result


def _trim_final_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    token_budget = int(context.get("token_budget") or DEFAULT_TOKEN_BUDGET)
    char_budget = max(MIN_TOKEN_BUDGET, token_budget) * 4
    trim_trace: list[str] = []

    if _estimated_chars(context) <= char_budget:
        context["trim_trace"] = trim_trace
        return context

    if context.get("related_long_term_memories"):
        context["related_long_term_memories"] = []
        trim_trace.append("drop_long_term_memory")

    if _estimated_chars(context) > char_budget and context.get("conversation_summary"):
        summary = context.get("conversation_summary")
        if isinstance(summary, dict):
            text = _summary_text(summary)
            context["conversation_summary"] = {**summary, "text": text[:500]}
        elif isinstance(summary, str):
            context["conversation_summary"] = summary[:500]
        trim_trace.append("truncate_conversation_summary")

    if _estimated_chars(context) > char_budget:
        context["recent_messages"] = _content_trimmed_messages(
            context.get("recent_messages") or [],
            8,
        )
        trim_trace.append("trim_recent_messages_to_8")

    if _estimated_chars(context) > char_budget and context.get("recent_turns"):
        context["recent_turns"] = (context.get("recent_turns") or [])[-4:]
        trim_trace.append("trim_recent_turns_to_4")

    if _estimated_chars(context) > char_budget and context.get("active_report"):
        report = context.get("active_report") or {}
        if isinstance(report, dict):
            context["active_report"] = _report_ref(report)
            trim_trace.append("reduce_active_report_to_ref")

    context["trim_trace"] = trim_trace
    return context


def _reference_target_from_context(
    *,
    query_context: dict[str, Any],
    last_answer: dict[str, Any] | None,
    active_report_ref: dict[str, Any] | None,
) -> dict[str, Any] | None:
    reference = query_context.get("reference_resolution") or {}
    if not isinstance(reference, dict):
        return None

    target_message_id = reference.get("target_message_id")
    if target_message_id and isinstance(last_answer, dict):
        return {
            "target_type": reference.get("target_type"),
            "target_message_id": target_message_id,
            "content": last_answer.get("content"),
            "answer_type": last_answer.get("answer_type"),
            "node_name": last_answer.get("node_name"),
            "report_id": last_answer.get("report_id"),
            "sources": last_answer.get("sources") or [],
        }

    target_report_id = reference.get("target_report_id")
    if target_report_id and isinstance(active_report_ref, dict):
        return {
            "target_type": reference.get("target_type"),
            "target_report_id": target_report_id,
            "active_report_ref": active_report_ref,
        }

    if reference.get("is_context_dependent") and isinstance(last_answer, dict):
        return {
            "target_type": reference.get("target_type"),
            "content": last_answer.get("content"),
            "answer_type": last_answer.get("answer_type"),
            "node_name": last_answer.get("node_name"),
            "sources": last_answer.get("sources") or [],
        }
    return None


def build_final_prompt_context(
    state: AgentState,
    *,
    system_prompt: str,
    node_name: str | None = None,
    include_long_term_memory: bool = False,
) -> dict[str, Any]:
    query_rewrite_context = query_rewrite_context_from_state(state)
    query_context = query_context_from_state(state)
    business_context = _business_context_from_state(state)
    active_report = (
        business_context.get("active_report")
        if isinstance(business_context.get("active_report"), dict)
        else None
    )
    active_report_ref = _report_ref(active_report) or query_rewrite_context.get("active_report_ref")
    last_answer = last_final_answer_from_state(state)

    final_context = _dump(
        FinalPromptContext(
            node_name=node_name,
            system_prompt=system_prompt,
            raw_user_input=query_rewrite_context.get("raw_user_input") or extract_user_text(state),
            standalone_query=query_context.get("standalone_query") or effective_user_query(state),
            reference_resolution=query_context.get("reference_resolution") or {},
            reference_target=_reference_target_from_context(
                query_context=query_context,
                last_answer=last_answer,
                active_report_ref=active_report_ref,
            ),
            intent_result=state.get("intent_result") or {},
            recent_messages=recent_messages_from_state(state),
            recent_turns=query_rewrite_context.get("recent_turns") or [],
            conversation_summary=query_rewrite_context.get("conversation_summary"),
            last_final_answer=last_answer,
            active_report=active_report or active_report_ref,
            active_report_ref=active_report_ref,
            business_context=business_context,
            privacy_policy=_privacy_policy_from_state(state),
            related_long_term_memories=(
                related_long_term_memories_from_state(state)
                if include_long_term_memory
                else []
            ),
            context_health=query_rewrite_context.get("context_health") or {},
            context_sources=query_rewrite_context.get("context_sources") or [],
            token_budget=token_budget_from_state(state),
        )
    )
    return _trim_final_prompt_context(final_context)


def build_prompt_context(
    state: AgentState,
    *,
    mode: ContextMode,
    system_prompt: str,
    node_name: str | None = None,
    include_long_term_memory: bool = False,
) -> dict[str, Any]:
    if mode == "MINIMAL_PRE_INTENT":
        context = query_rewrite_context_from_state(state)
        return {
            "schema_version": "1.0",
            "mode": mode,
            "node_name": node_name,
            "system_prompt": system_prompt,
            **context,
        }
    return build_final_prompt_context(
        state,
        system_prompt=system_prompt,
        node_name=node_name,
        include_long_term_memory=include_long_term_memory,
    )


def with_prompt_context(
    state: AgentState,
    *,
    mode: ContextMode,
    system_prompt: str,
    node_name: str | None = None,
    include_long_term_memory: bool = False,
) -> AgentState:
    current_turn = ensure_current_turn(state)
    if mode == "MINIMAL_PRE_INTENT":
        query_rewrite_context = build_query_rewrite_context(state)
        short_term_context = select_short_term_context(state)
        current_turn["short_term_context"] = short_term_context
        current_turn["query_rewrite_context"] = query_rewrite_context
        current_turn["context_health"] = query_rewrite_context.get("context_health") or {}
        prompt_context = {
            "schema_version": "1.0",
            "mode": mode,
            "node_name": node_name,
            "system_prompt": system_prompt,
            **query_rewrite_context,
        }
    else:
        prompt_context = build_final_prompt_context(
            {**state, "current_turn": current_turn},
            system_prompt=system_prompt,
            node_name=node_name,
            include_long_term_memory=include_long_term_memory,
        )
        current_turn["final_prompt_context"] = prompt_context

    return {
        **state,
        "current_turn": current_turn,
        "prompt_context": prompt_context,
    }
