"""Production-oriented Agent runtime subgraph compatibility boundary.

The runtime still reuses several legacy loop helpers. This module applies the
cross-cutting policies that must hold before those helpers are imported into the
compiled child graph: strict report binding, current-turn ownership and a safe
fallback that never returns a foreign checkpoint response.
"""

from typing import Any

from app.agent.nodes import agent_loop_node as legacy_agent_loop
from app.agent.state import AgentState
from app.agent.tool_policy import (
    can_use_report_followup,
    report_tool_rejection_observation,
)
from app.agent.turn_lifecycle import (
    build_current_turn_fallback,
    stamp_current_turn_artifacts,
)


REPORT_CONTEXT_TOOL = "report_context_tool"
REPORT_FOLLOWUP_TOOL = "report_followup_tool"
REPORT_BOUND_TOOLS = {REPORT_CONTEXT_TOOL, REPORT_FOLLOWUP_TOOL}



def _merge_next_action_compat(
    next_action: Any,
    agent_loop: dict[str, Any],
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Merge runtime metadata independently of the legacy helper signature."""

    turn_id = None
    request_id = None
    if isinstance(state, dict):
        turn_id = state.get("turn_id")
        request_id = state.get("request_id")
    turn_id = turn_id or agent_loop.get("turn_id")

    if not isinstance(next_action, dict):
        return {
            "type": "RESPOND_TO_USER",
            "turn_id": turn_id,
            "request_id": request_id,
            "payload": {
                "status": "COMPLETED",
                "agent_loop": agent_loop,
                "turn_id": turn_id,
                "request_id": request_id,
            },
        }

    payload = next_action.get("payload")
    payload = dict(payload) if isinstance(payload, dict) else {}
    return {
        **next_action,
        "turn_id": next_action.get("turn_id") or turn_id,
        "request_id": next_action.get("request_id") or request_id,
        "payload": {
            **payload,
            "agent_loop": agent_loop,
            "turn_id": payload.get("turn_id") or turn_id,
            "request_id": payload.get("request_id") or request_id,
        },
    }


# Preserve true legacy implementations across importlib.reload or test reloads.
_original_execute_tool_call = getattr(
    legacy_agent_loop,
    "_runtime_original_execute_tool_call",
    legacy_agent_loop._execute_tool_call,
)
_original_state_from_final_answer = getattr(
    legacy_agent_loop,
    "_runtime_original_state_from_final_answer",
    legacy_agent_loop._state_from_final_answer,
)
legacy_agent_loop._runtime_original_execute_tool_call = _original_execute_tool_call
legacy_agent_loop._runtime_original_state_from_final_answer = _original_state_from_final_answer


async def _execute_tool_call_guarded(
    *,
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[AgentState, dict[str, Any]]:
    """Apply deterministic policy before any report-bound tool executes."""

    if tool_name in REPORT_BOUND_TOOLS and not can_use_report_followup(state):
        return state, report_tool_rejection_observation(state, tool_name=tool_name)

    next_state, result = await _original_execute_tool_call(
        state=state,
        tool_name=tool_name,
        arguments=arguments,
    )
    next_state = stamp_current_turn_artifacts(next_state)
    result = dict(result)
    result.setdefault("turn_id", state.get("turn_id"))
    result.setdefault("request_id", state.get("request_id"))
    return next_state, result



def _state_from_final_answer_guarded(state: AgentState, content: str) -> AgentState:
    result = _original_state_from_final_answer(state, content)
    return stamp_current_turn_artifacts(result)



def _fallback_from_current_turn(state: AgentState) -> AgentState:
    response = state.get("response_message")
    if isinstance(response, dict):
        owner_turn_id = response.get("turn_id")
        current_turn_id = str(state.get("turn_id") or state.get("request_id") or "")
        if owner_turn_id is None or str(owner_turn_id) == current_turn_id:
            return stamp_current_turn_artifacts(state)

    return build_current_turn_fallback(
        state,
        error_code="INVALID_FINAL_RESPONSE",
        reason="current_turn_fallback_required",
    )


# Patch the compatibility seam before importing runtime.graph. The child graph
# therefore binds these guarded functions once during module import.
legacy_agent_loop._merge_agent_loop_into_next_action = _merge_next_action_compat
legacy_agent_loop._execute_tool_call = _execute_tool_call_guarded
legacy_agent_loop._state_from_final_answer = _state_from_final_answer_guarded
legacy_agent_loop._fallback_from_last_tool = _fallback_from_current_turn

from app.agent.runtime.graph import build_agent_runtime_subgraph  # noqa: E402

__all__ = ["build_agent_runtime_subgraph"]
