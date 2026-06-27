from typing import Any

from app.agent.context_builder import (
    effective_user_query,
    ensure_current_turn,
    last_final_answer_from_state,
    query_context_from_state,
    query_rewrite_context_from_state,
)
from app.agent.state import AgentState
from app.integrations.report_sections_client import load_report_sections_from_java


REPORT_TARGETS = {"ACTIVE_REPORT", "REPORT_ITEM"}
REPORT_ROUTES = {"report_followup_subgraph", "report_explanation_subgraph"}
VALID_MODES = {"AUTO", "NONE", "LAST_ANSWER", "ACTIVE_REPORT"}


async def report_context_resolver_node(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    business_context = dict(current_turn.get("business_context") or {})
    business_context.pop("active_report", None)
    business_context.pop("loaded_report_sections", None)
    business_context.pop("report_context_error", None)

    mode = _report_context_mode(state)
    ref = _active_report_ref(state)
    should_load, reason = _should_load_report(state, mode=mode, ref=ref)
    decision = {
        "schema_version": "1.0",
        "turn_id": state.get("turn_id") or state.get("request_id"),
        "mode": mode,
        "status": "SKIPPED",
        "reason": reason,
        "sections": [],
    }

    if not should_load:
        business_context["active_report_ref"] = ref
        business_context["report_context_resolution"] = decision
        current_turn["business_context"] = business_context
        return {
            **state,
            "current_node": "report_context_resolver_node",
            "current_turn": current_turn,
        }

    if not ref:
        decision.update({"status": "FAILED", "reason": "trusted_active_report_ref_missing"})
        business_context["report_context_error"] = decision
        business_context["report_context_resolution"] = decision
        current_turn["business_context"] = business_context
        return {
            **state,
            "current_node": "report_context_resolver_node",
            "current_turn": current_turn,
        }

    sections = _sections_for_query(effective_user_query(state))
    decision["sections"] = sections
    result = await load_report_sections_from_java(
        tenant_id=state.get("tenant_id"),
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id"),
        thread_epoch=state.get("thread_epoch"),
        conversation_id=state.get("conversation_id"),
        turn_id=state.get("turn_id") or state.get("request_id"),
        report_id=ref.get("report_id"),
        report_version=ref.get("report_version"),
        sections=sections,
    )

    loaded = _normalize_loaded_sections(result, ref=ref, sections=sections)
    if loaded is None:
        error_reason = result.get("error") or result.get("status") or "invalid_report_sections"
        if str(error_reason).upper() in {"OK", "SUCCESS", "COMPLETED"}:
            error_reason = "invalid_report_sections"
        decision.update(
            {
                "status": "FAILED",
                "reason": str(error_reason),
            }
        )
        business_context["active_report_ref"] = ref
        business_context["report_context_error"] = decision
        business_context["report_context_resolution"] = decision
        current_turn["business_context"] = business_context
        return {
            **state,
            "current_node": "report_context_resolver_node",
            "current_turn": current_turn,
        }

    decision.update({"status": "LOADED", "reason": "loaded_from_java"})
    business_context["active_report_ref"] = ref
    business_context["loaded_report_sections"] = loaded
    business_context["report_context_resolution"] = decision
    current_turn["business_context"] = business_context
    return {
        **state,
        "current_node": "report_context_resolver_node",
        "current_turn": current_turn,
    }


def _report_context_mode(state: AgentState) -> str:
    client_context = state.get("client_context") or {}
    extra = client_context.get("extra") if isinstance(client_context, dict) else {}
    value = None
    if isinstance(client_context, dict):
        value = client_context.get("report_context_mode")
    if value is None and isinstance(extra, dict):
        value = extra.get("report_context_mode")
    mode = str(value or "AUTO").upper()
    return mode if mode in VALID_MODES else "AUTO"


def _active_report_ref(state: AgentState) -> dict[str, Any] | None:
    context = query_rewrite_context_from_state(state)
    ref = context.get("active_report_ref") if isinstance(context, dict) else None
    return ref if isinstance(ref, dict) and ref.get("report_id") is not None else None


def _should_load_report(
    state: AgentState,
    *,
    mode: str,
    ref: dict[str, Any] | None,
) -> tuple[bool, str]:
    if mode == "NONE":
        return False, "mode_none"
    if mode == "ACTIVE_REPORT":
        return True, "mode_active_report"

    query_context = query_context_from_state(state)
    reference = query_context.get("reference_resolution") or {}
    target_type = str(reference.get("target_type") or "") if isinstance(reference, dict) else ""

    if mode == "LAST_ANSWER":
        last_answer = last_final_answer_from_state(state) or {}
        if ref and last_answer.get("report_id") is not None and str(last_answer.get("report_id")) == str(ref.get("report_id")):
            return True, "mode_last_answer_report"
        return False, "mode_last_answer_without_report"

    route_hint = str(query_context.get("route_hint") or "")
    intent_result = state.get("intent_result") or {}
    route_target = str(intent_result.get("route_target") or "")
    if target_type in REPORT_TARGETS:
        return True, "reference_targets_report"
    if route_hint in REPORT_ROUTES or route_target in REPORT_ROUTES:
        return True, "route_targets_report"
    return False, "auto_no_report_dependency"


def _sections_for_query(query: str) -> list[str]:
    compact = "".join(query.split())
    if any(keyword in compact for keyword in ("完整报告", "详细报告", "更详细的报告", "重新生成报告")):
        return ["full_report"]
    if any(keyword in compact for keyword in ("饮食", "吃什么", "怎么吃", "忌口", "食物")):
        return ["feature_summary", "interpretation", "dietary_advice"]
    if any(keyword in compact for keyword in ("运动", "锻炼", "跑步", "健身")):
        return ["feature_summary", "exercise_advice", "risk_disclaimer"]
    return ["feature_summary", "interpretation", "risk_disclaimer"]


def _normalize_loaded_sections(
    result: dict[str, Any],
    *,
    ref: dict[str, Any],
    sections: list[str],
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    status = str(result.get("status") or "OK").upper()
    if status not in {"OK", "SUCCESS", "COMPLETED"}:
        return None
    response_report_id = result.get("report_id", ref.get("report_id"))
    if str(response_report_id) != str(ref.get("report_id")):
        return None
    response_version = result.get("report_version", ref.get("report_version"))
    if ref.get("report_version") is not None and str(response_version) != str(ref.get("report_version")):
        return None

    raw_sections = result.get("sections")
    if raw_sections is None and isinstance(result.get("data"), dict):
        raw_sections = result["data"].get("sections")
    if not isinstance(raw_sections, dict) or not raw_sections:
        return None

    return {
        "schema_version": "1.0",
        "source": "java_report_sections",
        "report_id": ref.get("report_id"),
        "report_version": ref.get("report_version"),
        "feature_summary": _section_text(raw_sections.get("feature_summary")),
        "summary": _section_text(raw_sections.get("interpretation")),
        "requested_sections": sections,
        "sections": raw_sections,
    }


def _section_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("summary", "content", "text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        items = value.get("items")
        if isinstance(items, list):
            return "；".join(str(item).strip() for item in items if str(item).strip())
    if isinstance(value, list):
        return "；".join(str(item).strip() for item in value if str(item).strip())
    return ""
