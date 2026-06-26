from app.agent.state import AgentState
from app.agent.context_builder import active_report_from_state, query_context_from_state


GENERAL_CHAT_ROUTE = "general_chat_subgraph"
REPORT_FOLLOWUP_ROUTE = "report_followup_subgraph"
MVP_IMPLEMENTED_ROUTES = {
    "tongue_analysis_subgraph",
    "health_qa_subgraph",
    "privacy_request_subgraph",
    "high_risk_safety_subgraph",
    "general_chat_subgraph",
    "report_followup_subgraph",
}

ROUTE_TO_NODE = {
    "tongue_analysis_subgraph": "tongue_analysis_node",
    "health_qa_subgraph": "health_qa_node",
    "report_explanation_subgraph": "report_explanation_node",
    "report_followup_subgraph": "report_followup_node",
    "trend_analysis_subgraph": "trend_analysis_node",
    "privacy_request_subgraph": "privacy_request_node",
    "high_risk_safety_subgraph": "safety_node",
    "general_chat_subgraph": "general_chat_node",
}


def _normalize_route_target(route_target: str | None) -> str:
    if not route_target:
        return GENERAL_CHAT_ROUTE

    if route_target not in ROUTE_TO_NODE:
        return GENERAL_CHAT_ROUTE

    return route_target


def _resolve_next_node(route_target: str) -> tuple[str, bool]:
    if route_target not in MVP_IMPLEMENTED_ROUTES:
        return ROUTE_TO_NODE[GENERAL_CHAT_ROUTE], True

    return ROUTE_TO_NODE[route_target], False


async def route_node(state: AgentState) -> AgentState:
    intent_result = state.get("intent_result") or {}
    route_target = _resolve_route_target_from_state(state)
    next_node, mvp_fallback = _resolve_next_node(route_target)

    return {
        **state,
        "current_node": "route_node",
        "next_action": {
            "type": "ROUTE_TO_NODE",
            "payload": {
                "route_target": route_target,
                "next_node": next_node,
                "primary_intent": intent_result.get("primary_intent"),
                "detected_intent": intent_result.get("detected_intent"),
                "decision": intent_result.get("decision"),
                "confidence": intent_result.get("confidence"),
                "risk_level": intent_result.get("risk_level"),
                "mvp_fallback": mvp_fallback,
            },
        },
    }


def _resolve_route_target_from_state(state: AgentState) -> str:
    intent_result = state.get("intent_result") or {}
    route_target = _normalize_route_target(intent_result.get("route_target"))
    query_context = query_context_from_state(state)
    route_hint = query_context.get("route_hint")
    has_report_context = bool(active_report_from_state(state))

    if query_context.get("clarification_status") == "NEEDS_CLARIFICATION":
        return GENERAL_CHAT_ROUTE

    if route_target in {"high_risk_safety_subgraph", "privacy_request_subgraph"}:
        return route_target

    if isinstance(route_hint, str) and route_hint in ROUTE_TO_NODE:
        if route_hint != REPORT_FOLLOWUP_ROUTE or has_report_context:
            return route_hint

    if has_report_context and route_target == "report_explanation_subgraph":
        return REPORT_FOLLOWUP_ROUTE

    return route_target


def select_next_route(state: AgentState) -> str:
    route_target = _resolve_route_target_from_state(state)
    next_node, _ = _resolve_next_node(route_target)
    return next_node
