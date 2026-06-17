import asyncio
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.integrations.model_gateway import get_embedding_model
from app.intent.seed_intents import DOMAIN_TERMS, INTENT_DEFINITIONS, INTENT_EXAMPLES


def _definition_by_code() -> dict[str, dict[str, Any]]:
    return {item["intent_code"]: item for item in INTENT_DEFINITIONS}


def _intent_mapping(embedding_dim: int) -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "intent_code": {"type": "keyword"},
                "intent_name": {"type": "text", "analyzer": "standard"},
                "description": {"type": "text", "analyzer": "standard"},
                "route_target": {"type": "keyword"},
                "risk_level": {"type": "keyword"},
                "example_id": {"type": "keyword"},
                "example_text": {"type": "text", "analyzer": "standard"},
                "keywords": {"type": "text", "analyzer": "standard"},
                "required_slots": {"type": "keyword"},
                "optional_slots": {"type": "keyword"},
                "priority": {"type": "integer"},
                "weight": {"type": "float"},
                "source": {"type": "keyword"},
                "quality_score": {"type": "float"},
                "review_status": {"type": "keyword"},
                "enabled": {"type": "boolean"},
                "version": {"type": "integer"},
                "threshold": {"type": "object", "enabled": True},
                "clarify_template": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": False,
                },
            }
        }
    }


def _domain_term_mapping() -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "term_id": {"type": "keyword"},
                "raw_term": {"type": "text", "analyzer": "standard"},
                "normalized_term": {"type": "text", "analyzer": "standard"},
                "term_type": {"type": "keyword"},
                "aliases": {"type": "text", "analyzer": "standard"},
                "related_intents": {"type": "keyword"},
                "risk_level": {"type": "keyword"},
                "enabled": {"type": "boolean"},
                "version": {"type": "integer"},
            }
        }
    }


async def _recreate_index(
    es: AsyncElasticsearch,
    *,
    index_name: str,
    mapping: dict[str, Any],
) -> None:
    exists = await es.indices.exists(index=index_name)
    if exists:
        await es.indices.delete(index=index_name)
    await es.indices.create(index=index_name, **mapping)


async def _build_intent_documents() -> list[dict[str, Any]]:
    definitions = _definition_by_code()
    embedding_model = get_embedding_model()
    documents: list[dict[str, Any]] = []

    for group in INTENT_EXAMPLES:
        intent_code = group["intent_code"]
        definition = definitions[intent_code]
        examples = group["examples"]
        embeddings = await embedding_model.embed_texts(examples)

        for index, (example_text, embedding) in enumerate(zip(examples, embeddings), 1):
            documents.append(
                {
                    "intent_code": intent_code,
                    "intent_name": definition["intent_name"],
                    "description": definition["description"],
                    "route_target": definition["route_target"],
                    "risk_level": definition["risk_level"],
                    "example_id": f"{intent_code.lower()}_{index:03d}",
                    "example_text": example_text,
                    "keywords": group["keywords"],
                    "required_slots": definition["required_slots"],
                    "optional_slots": definition["optional_slots"],
                    "priority": definition["priority"],
                    "weight": 1.0,
                    "source": "seed_manual",
                    "quality_score": 1.0,
                    "review_status": "APPROVED",
                    "enabled": definition["enabled"],
                    "version": definition["version"],
                    "threshold": definition["threshold"],
                    "clarify_template": definition["clarify_template"],
                    "embedding": embedding,
                }
            )

    return documents


def _bulk_actions(index_name: str, documents: list[dict[str, Any]]):
    for document in documents:
        document_id = document.get("example_id") or document.get("term_id")
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": document_id,
            "_source": document,
        }


async def seed_intent_repository() -> None:
    settings = get_settings()
    es = create_es_client()

    try:
        await _recreate_index(
            es,
            index_name=settings.intent_index_name,
            mapping=_intent_mapping(settings.embedding_dim),
        )
        await _recreate_index(
            es,
            index_name=settings.domain_term_index_name,
            mapping=_domain_term_mapping(),
        )

        intent_documents = await _build_intent_documents()
        await async_bulk(
            es,
            _bulk_actions(settings.intent_index_name, intent_documents),
            refresh=True,
        )
        await async_bulk(
            es,
            _bulk_actions(settings.domain_term_index_name, DOMAIN_TERMS),
            refresh=True,
        )

        print(
            "Seeded intent repository: "
            f"{len(intent_documents)} intent examples, "
            f"{len(DOMAIN_TERMS)} domain terms."
        )
    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(seed_intent_repository())
