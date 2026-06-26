from __future__ import annotations

from typing import Any

from app.agent.context_builder import active_report_from_state, query_context_from_state
from app.agent.state import AgentState


REPORT_FOLLOWUP_ROUTE = "report_followup_subgraph"
REPORT_EXPLANATION_ROUTE = "report_explanation_subgraph"
REPORT_REFERENCE_TARGETS = {"ACTIVE_REPORT", "REPORT_ITEM"}



def _raw_user_input(state: AgentState) -> str:
    query_context = query_context_from_state(state)
    raw = query_context.get("raw_user_input")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    message = state.get("message") or {}
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""



def _explicit_report_reference(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    markers = (
        "报告",
        "上一份舌象",
        "上一次舌象",
        "刚才的舌象",
        "前面的舌象",
        "这份舌象",
        "这个舌象",
        "根据我的舌象",
        "结合我的舌象",
        "识别结果",
        "分析结果",
    )
    return any(marker in compact for marker in markers)



def can_use_report_followup(state: AgentState) -> bool:
    """Return whether this turn is explicitly allowed to bind an active report.

    Merely having an active report is insufficient. The query rewrite result or
    raw user text must establish a report reference, preventing unrelated health
    questions from being silently rewritten as report follow-ups.
    """

    if not active_report_from_state(state):
        return False

    query_context = query_context_from_state(state)
    route_hint = str(query_context.get("route_hint") or "")
    reference = query_context.get("reference_resolution") or {}
    target_type = (
        str(reference.get("target_type") or "")
        if isinstance(reference, dict)
        else ""
    )

    intent_result = state.get("intent_result") or {}
    route_target = str(intent_result.get("route_target") or "")

    if route_hint == REPORT_FOLLOWUP_ROUTE:
        return True
    if route_target == REPORT_EXPLANATION_ROUTE:
        return True
    if target_type in REPORT_REFERENCE_TARGETS:
        return True
    return _explicit_report_reference(_raw_user_input(state))



def report_tool_rejection_reason(state: AgentState) -> str:
    if not active_report_from_state(state):
        return "active_report_required"
    return "explicit_report_reference_required"



def report_tool_rejection_observation(
    state: AgentState,
    *,
    tool_name: str,
) -> dict[str, Any]:
    return {
        "status": "REJECTED",
        "tool_name": tool_name,
        "error": report_tool_rejection_reason(state),
        "retryable": True,
        "recommended_tool": "health_qa_tool",
        "policy": "strict_report_binding.v1",
        "turn_id": state.get("turn_id"),
        "request_id": state.get("request_id"),
    }
