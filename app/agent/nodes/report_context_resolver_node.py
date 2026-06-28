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
ALLOWED_REPORT_SECTIONS = {
    "feature_summary",
    "interpretation",
    "dietary_advice",
    "exercise_advice",
    "lifestyle_advice",
    "risk_disclaimer",
    "rag_evidence_summary",
    "full_report",
}


async def report_context_resolver_node(state: AgentState) -> AgentState:
    current_turn = ensure_current_turn(state)
    business_context = dict(current_turn.get("business_context") or {})
    for key in (
        "active_report",
        "loaded_report_sections",
        "report_context_error",
        "report_context_resolution",
    ):
        business_context.pop(key, None)

    mode = _report_context_mode(state)
    ref = _active_report_ref(state)
    report_load_plan = _report_load_plan_from_state(state)
    should_load, reason = _should_load_report(
        state,
        mode=mode,
        ref=ref,
        report_load_plan=report_load_plan,
    )
    decision = {
        "schema_version": "1.0",
        "turn_id": state.get("turn_id") or state.get("request_id"),
        "mode": mode,
        "report_id": ref.get("report_id") if ref else None,
        "status": "SKIPPED",
        "reason": reason,
        "sections": [],
    }
    if report_load_plan is not None:
        decision["report_load_plan"] = {
            "need_report": bool(report_load_plan.get("need_report")),
            "target_report_id": report_load_plan.get("target_report_id"),
            "target_type": report_load_plan.get("target_type"),
            "target_focus": report_load_plan.get("target_focus"),
            "sections": report_load_plan.get("sections") or [],
            "confidence": report_load_plan.get("confidence"),
        }

    if not should_load:
        return _finish(state, current_turn, business_context, ref, decision)

    if not ref:
        decision.update(status="FAILED", reason="trusted_active_report_ref_missing")
        return _finish(state, current_turn, business_context, ref, decision, error=True)

    if not _is_trusted_ref(ref):
        decision.update(status="FAILED", reason="untrusted_active_report_ref")
        return _finish(state, current_turn, business_context, ref, decision, error=True)

    if mode == "ACTIVE_REPORT" and not _matches_requested_report(state, ref):
        decision.update(status="FAILED", reason="active_report_id_mismatch")
        return _finish(state, current_turn, business_context, ref, decision, error=True)

    if report_load_plan is not None and not _report_plan_matches_ref(report_load_plan, ref):
        decision.update(status="FAILED", reason="report_load_plan_report_id_mismatch")
        return _finish(state, current_turn, business_context, ref, decision, error=True)

    sections = _sections_from_report_load_plan(report_load_plan)
    if report_load_plan is not None and report_load_plan.get("need_report") and not sections:
        decision.update(status="FAILED", reason="invalid_report_load_plan_sections")
        return _finish(state, current_turn, business_context, ref, decision, error=True)
    if not sections:
        sections = _sections_for_query(effective_user_query(state))
    decision["sections"] = sections
    preloaded = _preloaded_sections(state, sections=sections)
    result = preloaded or await _load_sections(state, ref=ref, sections=sections)

    if str(result.get("status") or "").upper() == "VERSION_MISMATCH":
        current_version = result.get("report_version")
        if current_version is not None:
            refreshed_ref = {**ref, "report_version": current_version}
            retry_result = await _load_sections(
                state,
                ref=refreshed_ref,
                sections=sections,
            )
            if str(retry_result.get("status") or "").upper() in {
                "OK",
                "SUCCESS",
                "COMPLETED",
            }:
                ref = refreshed_ref
                result = retry_result
                decision["report_version_refreshed"] = True

    loaded = _normalize_loaded_sections(result, ref=ref, sections=sections)
    if loaded is None:
        error_reason = result.get("error") or result.get("status") or "invalid_report_sections"
        if str(error_reason).upper() in {"OK", "SUCCESS", "COMPLETED"}:
            error_reason = "invalid_report_sections"
        decision.update(status="FAILED", reason=str(error_reason))
        return _finish(state, current_turn, business_context, ref, decision, error=True)

    decision.update(
        status="LOADED",
        reason="loaded_from_context_bundle" if preloaded else "loaded_from_java",
        report_version=loaded.get("report_version"),
    )
    business_context["loaded_report_sections"] = loaded
    return _finish(state, current_turn, business_context, ref, decision)


def _preloaded_sections(state: AgentState, *, sections: list[str]) -> dict[str, Any] | None:
    bundle = state.get("context_bundle") or {}
    if not isinstance(bundle, dict):
        return None
    loaded = bundle.get("loaded_report_sections")
    if not isinstance(loaded, dict) or not loaded:
        return None
    raw_sections = loaded.get("sections")
    if raw_sections is None and isinstance(loaded.get("data"), dict):
        raw_sections = loaded["data"].get("sections")
    if not isinstance(raw_sections, dict):
        return None
    return loaded if all(section in raw_sections for section in sections) else None


async def _load_sections(
    state: AgentState,
    *,
    ref: dict[str, Any],
    sections: list[str],
) -> dict[str, Any]:
    return await load_report_sections_from_java(
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


def _finish(
    state: AgentState,
    current_turn: dict[str, Any],
    business_context: dict[str, Any],
    ref: dict[str, Any] | None,
    decision: dict[str, Any],
    *,
    error: bool = False,
) -> AgentState:
    business_context["active_report_ref"] = ref
    business_context["report_context_resolution"] = decision
    if error:
        business_context["report_context_error"] = decision
    current_turn["business_context"] = business_context
    intent_result = dict(state.get("intent_result") or {})
    intent_result["report_context"] = decision
    return {
        **state,
        "current_node": "report_context_resolver_node",
        "current_turn": current_turn,
        "intent_result": intent_result,
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


def _is_trusted_ref(ref: dict[str, Any]) -> bool:
    return ref.get("trusted") is True or ref.get("is_current_active_report") is True


def _matches_requested_report(state: AgentState, ref: dict[str, Any]) -> bool:
    client_context = state.get("client_context") or {}
    requested = client_context.get("active_report_id") if isinstance(client_context, dict) else None
    if requested is None:
        return False
    return str(requested) == str(ref.get("report_id"))


def _report_load_plan_from_state(state: AgentState) -> dict[str, Any] | None:
    query_context = query_context_from_state(state)
    plan = query_context.get("report_load_plan") if isinstance(query_context, dict) else None
    return plan if isinstance(plan, dict) else None


def _report_plan_matches_ref(plan: dict[str, Any], ref: dict[str, Any]) -> bool:
    target_report_id = plan.get("target_report_id")
    return target_report_id is None or str(target_report_id) == str(ref.get("report_id"))


def _sections_from_report_load_plan(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict) or not plan.get("need_report"):
        return []
    sections = plan.get("sections") or []
    if not isinstance(sections, list):
        return []
    deduped = list(dict.fromkeys(str(section) for section in sections))
    if not deduped or any(section not in ALLOWED_REPORT_SECTIONS for section in deduped):
        return []
    return deduped


def _should_load_report(
    state: AgentState,
    *,
    mode: str,
    ref: dict[str, Any] | None,
    report_load_plan: dict[str, Any] | None,
) -> tuple[bool, str]:
    if mode == "NONE":
        return False, "mode_none"
    if mode == "ACTIVE_REPORT":
        return True, "mode_active_report"
    if isinstance(report_load_plan, dict):
        if report_load_plan.get("need_report"):
            return True, "query_rewrite_report_load_plan"
        return False, "query_rewrite_plan_no_report_dependency"

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
    asks_for_report = "报告" in compact and any(
        keyword in compact
        for keyword in ("详细", "完整", "展开", "重新生成", "太简单", "不够详细")
    )
    if asks_for_report:
        return [
            "full_report",
            "feature_summary",
            "interpretation",
            "dietary_advice",
            "exercise_advice",
            "lifestyle_advice",
            "risk_disclaimer",
            "rag_evidence_summary",
        ]
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
        "report_version": response_version,
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
