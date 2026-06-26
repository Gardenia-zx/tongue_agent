from app.agent.context_builder import query_context_from_state
from app.agent.state import AgentState


AGENT_LOOP_NODE = "agent_loop_node"
GENERAL_CHAT_NODE = "general_chat_node"
SAFETY_NODE = "safety_node"
PRIVACY_NODE = "privacy_request_node"


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

    if clarification_status == "NEEDS_CLARIFICATION":
        return GENERAL_CHAT_NODE

    if route_hint:
        return AGENT_LOOP_NODE

    if route_target == "tongue_analysis_subgraph":
        return AGENT_LOOP_NODE

    return AGENT_LOOP_NODE
