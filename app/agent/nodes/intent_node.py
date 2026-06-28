import re
from typing import Any

from app.agent.context_builder import (
    active_report_from_state,
    effective_user_query,
    query_context_from_state,
)
from app.agent.state import AgentState
from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.intent.domain_terms import DomainTermNormalizer
from app.intent.es_intent_retriever import ESIntentRetriever


REPORT_ROUTE = "report_followup_subgraph"
DIET_OR_CARE_KEYWORDS = [
    "饮食",
    "吃什么",
    "怎么吃",
    "每天吃",
    "食物",
    "食谱",
    "忌口",
    "早餐",
    "午餐",
    "晚餐",
    "清淡",
    "生冷",
    "油腻",
    "甜腻",
    "调理",
    "规划",
]
EXPLICIT_DIAGNOSIS_PATTERN = re.compile(
    r"确诊|诊断|判断.*是不是|是不是.*病|得了什么病"
)
MEDICATION_OR_EMERGENCY_PATTERN = re.compile(
    r"药|处方|剂量|停药|换药|加药|减药|胸痛|呼吸困难|昏迷|大出血|休克|意识不清|抽搐"
)


def _get_embedding_model() -> Any:
    from app.integrations.model_gateway import get_embedding_model

    return get_embedding_model()


def _build_intent_service(
    *,
    retriever: Any,
    embedding_model: Any,
    domain_normalizer: Any,
) -> Any:
    from app.intent.service import IntentRecognitionService

    return IntentRecognitionService(
        retriever=retriever,
        embedding_model=embedding_model,
        domain_normalizer=domain_normalizer,
    )


def _extract_user_text(state: AgentState) -> str:
    message: dict[str, Any] = state.get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    return ""


def _build_next_action(intent_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "ROUTE_TO_SUBGRAPH",
        "payload": {
            "primary_intent": intent_result.get("primary_intent"),
            "detected_intent": intent_result.get("detected_intent"),
            "decision": intent_result.get("decision"),
            "route_target": intent_result.get("route_target"),
            "confidence": intent_result.get("confidence"),
            "risk_level": intent_result.get("risk_level"),
        },
    }


def _clarification_intent_result() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "engine": "QUERY_REWRITE_GATE",
        "engine_version": "query-rewrite-v1",
        "detected_intent": "CLARIFICATION",
        "primary_intent": "CLARIFICATION",
        "secondary_intents": [],
        "confidence": 1.0,
        "decision": "CLARIFY",
        "route_target": "general_chat_subgraph",
        "risk_level": "LOW",
        "missing_slots": ["reference_context"],
        "entities": [],
        "topk_candidates": [],
        "safety_flags": [],
        "debug": {"reason": "query_rewrite_needs_clarification"},
    }


def _canonicalize_report_followup(
    intent_result: dict[str, Any],
    *,
    query_context: dict[str, Any],
) -> dict[str, Any]:
    adjusted = dict(intent_result)
    adjusted.update(
        {
            "detected_intent": "REPORT_EXPLANATION",
            "primary_intent": "REPORT_EXPLANATION",
            "secondary_intents": [],
            "confidence": max(float(adjusted.get("confidence") or 0.0), 0.98),
            "decision": "ROUTE",
            "route_target": REPORT_ROUTE,
            "risk_level": "LOW",
            "missing_slots": [],
            "safety_flags": [],
        }
    )
    debug = dict(adjusted.get("debug") or {})
    debug["query_rewrite_route_hint"] = REPORT_ROUTE
    debug["query_rewrite_reference"] = query_context.get("reference_resolution") or {}
    debug["intent_override_reason"] = "trusted_report_reference_resolved"
    adjusted["debug"] = debug
    return adjusted


def _has_report_context(state: AgentState, query_context: dict[str, Any]) -> bool:
    if active_report_from_state(state):
        return True
    plan = query_context.get("report_load_plan")
    return (
        isinstance(plan, dict)
        and plan.get("need_report") is True
        and plan.get("target_report_id") is not None
    )


def _is_safe_diet_or_care_request(text: str) -> bool:
    if EXPLICIT_DIAGNOSIS_PATTERN.search(text) or MEDICATION_OR_EMERGENCY_PATTERN.search(text):
        return False
    return any(keyword in text for keyword in DIET_OR_CARE_KEYWORDS)


def _canonicalize_health_qa(
    intent_result: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    adjusted = dict(intent_result)
    adjusted.update(
        {
            "detected_intent": "HEALTH_QA",
            "primary_intent": "HEALTH_QA",
            "secondary_intents": [],
            "confidence": max(float(adjusted.get("confidence") or 0.0), 0.9),
            "decision": "ROUTE",
            "route_target": "health_qa_subgraph",
            "risk_level": "LOW",
            "missing_slots": [],
            "safety_flags": [],
        }
    )
    debug = dict(adjusted.get("debug") or {})
    debug["intent_override_reason"] = reason
    adjusted["debug"] = debug
    return adjusted


def _is_report_diet_or_care_followup(
    state: AgentState,
    query_context: dict[str, Any],
) -> bool:
    if not _has_report_context(state, query_context):
        return False

    reference = query_context.get("reference_resolution") or {}
    target_focus = str(reference.get("target_focus") or "")
    text = " ".join(
        str(query_context.get(key) or "")
        for key in ("raw_user_input", "standalone_query")
    )
    if EXPLICIT_DIAGNOSIS_PATTERN.search(text):
        return False
    return target_focus == "DIET_ADVICE" or any(
        keyword in text for keyword in DIET_OR_CARE_KEYWORDS
    )


def _apply_route_hint(
    intent_result: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    query_context = query_context_from_state(state)
    route_hint = query_context.get("route_hint")

    current_route = str(intent_result.get("route_target") or "")
    risk_level = str(intent_result.get("risk_level") or "")
    if current_route == "privacy_request_subgraph" or risk_level == "EMERGENCY":
        return intent_result

    if route_hint == REPORT_ROUTE:
        if not _has_report_context(state, query_context):
            text = " ".join(str(query_context.get(key) or "") for key in ("raw_user_input", "standalone_query"))
            if current_route == "high_risk_safety_subgraph" and _is_safe_diet_or_care_request(text):
                return _canonicalize_health_qa(intent_result, reason="safe_diet_request_without_report_context")
            return intent_result
        is_high_risk_route = current_route == "high_risk_safety_subgraph"
        if is_high_risk_route and not _is_report_diet_or_care_followup(
            state,
            query_context,
        ):
            return intent_result
        return _canonicalize_report_followup(
            intent_result,
            query_context=query_context,
        )

    text = " ".join(
        str(query_context.get(key) or "")
        for key in ("raw_user_input", "standalone_query")
    )
    if current_route == "high_risk_safety_subgraph" and _is_safe_diet_or_care_request(text):
        return _canonicalize_health_qa(intent_result, reason="safe_diet_request_false_positive")

    if route_hint not in {
        REPORT_ROUTE,
        "health_qa_subgraph",
        "general_chat_subgraph",
        "tongue_analysis_subgraph",
    }:
        return intent_result

    adjusted = dict(intent_result)
    adjusted["route_target"] = route_hint
    debug = dict(adjusted.get("debug") or {})
    debug["query_rewrite_route_hint"] = route_hint
    debug["query_rewrite_reference"] = query_context.get("reference_resolution") or {}
    adjusted["debug"] = debug
    return adjusted


async def intent_node(state: AgentState) -> AgentState:
    settings = get_settings()
    query_context = query_context_from_state(state)
    if query_context.get("clarification_status") == "NEEDS_CLARIFICATION":
        intent_result_dict = _clarification_intent_result()
        return {
            **state,
            "current_node": "intent_node",
            "intent_result": intent_result_dict,
            "next_action": _build_next_action(intent_result_dict),
        }

    query = str(
        query_context.get("standalone_query")
        or effective_user_query(state)
        or _extract_user_text(state)
    ).strip()

    es = create_es_client()
    try:
        retriever = ESIntentRetriever(
            es,
            index_name=settings.intent_index_name,
        )
        domain_normalizer = DomainTermNormalizer(
            es,
            index_name=settings.domain_term_index_name,
        )
        service = _build_intent_service(
            retriever=retriever,
            embedding_model=_get_embedding_model(),
            domain_normalizer=domain_normalizer,
        )

        intent_result = await service.recognize(query)
        intent_result_dict = _apply_route_hint(
            intent_result.model_dump(mode="json"),
            state,
        )

        return {
            **state,
            "current_node": "intent_node",
            "intent_result": intent_result_dict,
            "next_action": _build_next_action(intent_result_dict),
        }
    finally:
        await es.close()
