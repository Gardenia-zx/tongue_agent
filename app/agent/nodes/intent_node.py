from typing import Any

from app.agent.context_builder import active_report_from_state, effective_user_query, query_context_from_state
from app.agent.state import AgentState
from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.intent.domain_terms import DomainTermNormalizer
from app.intent.es_intent_retriever import ESIntentRetriever


REPORT_ROUTE = "report_followup_subgraph"


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
        }
    )
    debug = dict(adjusted.get("debug") or {})
    debug["query_rewrite_route_hint"] = REPORT_ROUTE
    debug["query_rewrite_reference"] = query_context.get("reference_resolution") or {}
    debug["intent_override_reason"] = "trusted_report_reference_resolved"
    adjusted["debug"] = debug
    return adjusted


def _apply_route_hint(
    intent_result: dict[str, Any],
    state: AgentState,
) -> dict[str, Any]:
    query_context = query_context_from_state(state)
    route_hint = query_context.get("route_hint")
    if route_hint not in {
        REPORT_ROUTE,
        "health_qa_subgraph",
        "general_chat_subgraph",
        "tongue_analysis_subgraph",
    }:
        return intent_result

    current_route = str(intent_result.get("route_target") or "")
    if current_route in {"high_risk_safety_subgraph", "privacy_request_subgraph"}:
        return intent_result

    if route_hint == REPORT_ROUTE:
        if not active_report_from_state(state):
            return intent_result
        return _canonicalize_report_followup(
            intent_result,
            query_context=query_context,
        )

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
