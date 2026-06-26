from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from app.agent.state import AgentState
from app.agent.tool_policy import can_use_report_followup


TURN_LOCAL_RESET_FIELDS = (
    "response_message",
    "next_action",
    "tool_decision",
    "quality_review",
    "rag_context",
    "tongue_features",
    "draft_report",
    "agent_loop",
    "safety_result",
    "intent_result",
    "query_context",
    "prompt_context",
    "current_turn",
    "image_quality",
    "retry",
    "memory_context",
)

REPORT_CONTEXT_KEYS = (
    "active_report",
    "active_report_ref",
    "latest_report",
    "latest_report_context",
    "frontend_latest_report",
)

HISTORY_CONTEXT_KEYS = (
    "recent_messages",
    "recent_turns",
    "last_final_answer",
    "conversation_summary",
    "traceback_context",
)



def _now_iso() -> str:
    return datetime.now(UTC).isoformat()



def _current_turn_id(state: AgentState) -> str:
    return str(state.get("turn_id") or state.get("request_id") or "")



def _current_request_id(state: AgentState) -> str:
    return str(state.get("request_id") or "")



def _artifact_owned_by_current_turn(artifact: Any, state: AgentState) -> bool:
    if not isinstance(artifact, dict):
        return False
    owner_turn_id = artifact.get("turn_id")
    return owner_turn_id is None or str(owner_turn_id) == _current_turn_id(state)



def _stamp_mapping(mapping: dict[str, Any], state: AgentState) -> dict[str, Any]:
    return {
        **mapping,
        "turn_id": _current_turn_id(state),
        "request_id": _current_request_id(state),
    }



def stamp_current_turn_artifacts(state: AgentState) -> AgentState:
    """Stamp artifacts created during this turn without accepting foreign owners."""

    next_state: AgentState = {**state}
    response = state.get("response_message")
    if isinstance(response, dict) and _artifact_owned_by_current_turn(response, state):
        next_state["response_message"] = _stamp_mapping(dict(response), state)

    next_action = state.get("next_action")
    if isinstance(next_action, dict) and _artifact_owned_by_current_turn(next_action, state):
        payload = next_action.get("payload")
        payload = dict(payload) if isinstance(payload, dict) else {}
        next_state["next_action"] = _stamp_mapping(
            {
                **next_action,
                "payload": _stamp_mapping(payload, state),
            },
            state,
        )

    return next_state



def build_current_turn_fallback(
    state: AgentState,
    *,
    error_code: str,
    reason: str,
) -> AgentState:
    """Build a deterministic response that can never reuse a previous turn."""

    errors = list(state.get("errors") or [])
    errors.append(
        {
            "code": error_code,
            "reason": reason,
            "turn_id": _current_turn_id(state),
            "request_id": _current_request_id(state),
            "node": state.get("current_node"),
        }
    )

    agent_loop = dict(state.get("agent_loop") or {})
    if agent_loop:
        agent_loop.update(
            {
                "status": "COMPLETED",
                "execution_status": "DEGRADED",
                "finish_reason": reason,
                "turn_id": _current_turn_id(state),
            }
        )

    response_message = {
        "role": "assistant",
        "content_type": "text",
        "content": "当前请求暂时未能生成有效回答，请重新描述问题或稍后再试。",
        "turn_id": _current_turn_id(state),
        "request_id": _current_request_id(state),
        "error_code": error_code,
    }
    next_action = {
        "type": "RESPOND_TO_USER",
        "turn_id": _current_turn_id(state),
        "request_id": _current_request_id(state),
        "payload": {
            "status": "FAILED",
            "answer_type": "FALLBACK",
            "error_code": error_code,
            "turn_id": _current_turn_id(state),
            "request_id": _current_request_id(state),
        },
    }
    return {
        **state,
        "response_message": response_message,
        "next_action": next_action,
        "agent_loop": agent_loop,
        "errors": errors,
    }


async def begin_turn_node(state: AgentState) -> AgentState:
    """Create a clean per-turn execution boundary on top of checkpointed state.

    The reset is applied only when the incoming turn differs from the turn stored
    in the previous checkpoint. Retrying or resuming the same turn therefore does
    not destroy its in-flight state.
    """

    current_turn_id = _current_turn_id(state)
    previous_guard = state.get("turn_guard") or {}
    previous_turn_id = (
        str(previous_guard.get("turn_id") or "")
        if isinstance(previous_guard, dict)
        else ""
    )
    reset_applied = previous_turn_id != current_turn_id

    next_state: AgentState = {**state}
    if reset_applied:
        for field in TURN_LOCAL_RESET_FIELDS:
            next_state[field] = None  # type: ignore[literal-required]
        next_state["errors"] = []

    extensions = dict(state.get("extensions") or {})
    extensions["turn_boundary"] = {
        "previous_turn_id": previous_turn_id or None,
        "current_turn_id": current_turn_id,
        "reset_applied": reset_applied,
    }

    next_state.update(
        {
            "current_node": "begin_turn_node",
            "extensions": extensions,
            "turn_guard": {
                "schema_version": "1.0",
                "turn_id": current_turn_id,
                "request_id": _current_request_id(state),
                "status": "STARTED",
                "reset_applied": reset_applied,
                "started_at": _now_iso(),
            },
        }
    )
    return next_state



def _drop_keys(value: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    for key in keys:
        result.pop(key, None)
    return result



def _sanitize_container(
    value: Any,
    *,
    allow_report: bool,
    allow_history: bool,
) -> Any:
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    if not allow_report:
        for key in REPORT_CONTEXT_KEYS:
            result.pop(key, None)
        result.pop("reference_target", None)
    if not allow_history:
        for key in HISTORY_CONTEXT_KEYS:
            if key in {"recent_messages", "recent_turns"}:
                result[key] = []
            else:
                result.pop(key, None)
    return result


async def runtime_context_policy_node(state: AgentState) -> AgentState:
    """Apply progressive disclosure before the model/tool runtime starts."""

    query_context = state.get("query_context") or {}
    reference = (
        query_context.get("reference_resolution") or {}
        if isinstance(query_context, dict)
        else {}
    )
    allow_history = bool(
        isinstance(reference, dict) and reference.get("is_context_dependent")
    )
    allow_report = can_use_report_followup(state)

    current_turn = deepcopy(state.get("current_turn") or {})
    if isinstance(current_turn, dict):
        for name in (
            "short_term_context",
            "query_rewrite_context",
            "business_context",
            "final_prompt_context",
        ):
            current_turn[name] = _sanitize_container(
                current_turn.get(name) or {},
                allow_report=allow_report,
                allow_history=allow_history,
            )

    prompt_context = _sanitize_container(
        state.get("prompt_context") or {},
        allow_report=allow_report,
        allow_history=allow_history,
    )
    context_bundle = _sanitize_container(
        state.get("context_bundle") or {},
        allow_report=allow_report,
        allow_history=allow_history,
    )

    client_context = deepcopy(state.get("client_context") or {})
    if isinstance(client_context, dict):
        extra = client_context.get("extra")
        if isinstance(extra, dict):
            extra = _sanitize_container(
                extra,
                allow_report=allow_report,
                allow_history=allow_history,
            )
            client_context["extra"] = extra

    policy = {
        "schema_version": "1.0",
        "turn_id": _current_turn_id(state),
        "allow_report_context": allow_report,
        "allow_conversation_history": allow_history,
        "reference_target_type": (
            reference.get("target_type") if isinstance(reference, dict) else None
        ),
        "route_hint": (
            query_context.get("route_hint") if isinstance(query_context, dict) else None
        ),
    }
    extensions = dict(state.get("extensions") or {})
    extensions["runtime_context_policy"] = policy

    return {
        **state,
        "current_node": "runtime_context_policy_node",
        "current_turn": current_turn,
        "prompt_context": prompt_context,
        "context_bundle": context_bundle,
        "client_context": client_context,
        "context_policy": policy,
        "extensions": extensions,
    }


async def final_response_guard_node(state: AgentState) -> AgentState:
    """Enforce final response ownership and repair invalid terminal states."""

    response = state.get("response_message")
    next_action = state.get("next_action")
    error_code: str | None = None
    reason: str | None = None

    if not isinstance(response, dict):
        error_code = "INVALID_FINAL_RESPONSE"
        reason = "response_message_missing"
    elif not _artifact_owned_by_current_turn(response, state):
        error_code = "STALE_RESPONSE_DETECTED"
        reason = "response_turn_id_mismatch"
    elif not str(response.get("content") or "").strip():
        error_code = "INVALID_FINAL_RESPONSE"
        reason = "response_content_empty"
    elif isinstance(next_action, dict) and not _artifact_owned_by_current_turn(
        next_action, state
    ):
        error_code = "TURN_OWNERSHIP_MISMATCH"
        reason = "next_action_turn_id_mismatch"

    if error_code and reason:
        guarded = build_current_turn_fallback(
            state,
            error_code=error_code,
            reason=reason,
        )
        guard_status = "REPAIRED"
    else:
        guarded = stamp_current_turn_artifacts(state)
        guard_status = "PASSED"

    turn_guard = dict(guarded.get("turn_guard") or {})
    turn_guard.update(
        {
            "turn_id": _current_turn_id(state),
            "request_id": _current_request_id(state),
            "status": guard_status,
            "completed_at": _now_iso(),
            "error_code": error_code,
        }
    )
    extensions = dict(guarded.get("extensions") or {})
    extensions["response_guard"] = {
        "status": guard_status,
        "error_code": error_code,
        "reason": reason,
        "source_node": state.get("current_node"),
    }

    return {
        **guarded,
        "turn_guard": turn_guard,
        "extensions": extensions,
    }
