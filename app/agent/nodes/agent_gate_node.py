from typing import Any

from app.agent.context_builder import query_context_from_state
from app.agent.state import AgentState


AGENT_LOOP_NODE = "agent_loop_node"
GENERAL_CHAT_NODE = "general_chat_node"
SAFETY_NODE = "safety_node"
PRIVACY_NODE = "privacy_request_node"
TONGUE_ANALYSIS_NODE = "tongue_analysis_node"


def _client_extra(state: AgentState) -> dict[str, Any]:
    client_context = state.get("client_context") or {}
    if not isinstance(client_context, dict):
        return {}
    extra = client_context.get("extra") or {}
    return extra if isinstance(extra, dict) else {}


def _has_tongue_image_input(state: AgentState) -> bool:
    message = state.get("message") or {}
    attachments = message.get("attachments") or []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        if attachment.get("file_type") == "image" or attachment.get("purpose") == "tongue_image":
            return True

    extra = _client_extra(state)
    return any(
        isinstance(extra.get(key), str) and str(extra.get(key)).strip()
        for key in (
            "tongue_image_path",
            "image_path",
            "tongue_image_url",
            "image_url",
        )
    )


def _is_dedicated_tongue_analysis_request(state: AgentState) -> bool:
    client_context = state.get("client_context") or {}
    page = str(client_context.get("page") or "") if isinstance(client_context, dict) else ""
    return page == "tongue_analyze" and _has_tongue_image_input(state)


async def agent_gate_node(state: AgentState) -> AgentState:
    next_node = select_agent_gate_next(state)
    intent_result = state.get("intent_result") or {}
    return {
        **state,
        "current_node": "agent_gate_node",
        "next_action": {
            "type": "AGENT_GATE_DECISION",
            "payload": {
                "status": "COMPLETED",
                "next_node": next_node,
                "route_hint": intent_result.get("route_target"),
                "risk_level": intent_result.get("risk_level"),
                "query_context": query_context_from_state(state),
                "has_tongue_image": _has_tongue_image_input(state),
            },
        },
    }


def select_agent_gate_next(state: AgentState) -> str:
    intent_result = state.get("intent_result") or {}
    route_target = str(intent_result.get("route_target") or "")
    risk_level = str(intent_result.get("risk_level") or "")
    query_context = query_context_from_state(state)
    route_hint = str(query_context.get("route_hint") or "")
    clarification_status = str(query_context.get("clarification_status") or "")

    if route_target == "high_risk_safety_subgraph" or risk_level in {"HIGH", "EMERGENCY"}:
        return SAFETY_NODE

    if route_target == "privacy_request_subgraph":
        return PRIVACY_NODE

    # The dedicated image-analysis API already performed file validation and
    # supplied a concrete path or URL. Do not ask the planner whether to use the
    # image tool; execute the deterministic image -> report workflow directly.
    if _is_dedicated_tongue_analysis_request(state):
        return TONGUE_ANALYSIS_NODE

    if clarification_status == "NEEDS_CLARIFICATION":
        return GENERAL_CHAT_NODE

    if route_hint:
        return AGENT_LOOP_NODE

    if route_target == "tongue_analysis_subgraph":
        return AGENT_LOOP_NODE

    return AGENT_LOOP_NODE
