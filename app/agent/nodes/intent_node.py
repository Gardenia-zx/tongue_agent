from typing import Any

from app.agent.context_builder import effective_user_query
from app.agent.state import AgentState
from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.integrations.model_gateway import get_embedding_model
from app.intent.domain_terms import DomainTermNormalizer
from app.intent.es_intent_retriever import ESIntentRetriever
from app.intent.service import IntentRecognitionService


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


async def intent_node(state: AgentState) -> AgentState:
    settings = get_settings()
    query = effective_user_query(state) or _extract_user_text(state)

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
        service = IntentRecognitionService(
            retriever=retriever,
            embedding_model=get_embedding_model(),
            domain_normalizer=domain_normalizer,
        )

        intent_result = await service.recognize(query)
        intent_result_dict = intent_result.model_dump(mode="json")

        return {
            **state,
            "current_node": "intent_node",
            "intent_result": intent_result_dict,
            "next_action": _build_next_action(intent_result_dict),
        }
    finally:
        await es.close()
