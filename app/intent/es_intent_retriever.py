from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch


@dataclass(frozen=True)
class RawIntentHit:
    intent_code: str
    route_target: str | None
    risk_level: str
    example_text: str | None
    keywords: list[str]
    embedding: list[float] | None
    matched_fields: list[str]
    bm25_score: float
    metadata: dict[str, Any]


class ESIntentRetriever:
    def __init__(
        self,
        es: AsyncElasticsearch,
        *,
        index_name: str,
    ) -> None:
        self.es = es
        self.index_name = index_name

    async def search(
        self,
        *,
        query: str,
        top_k: int,
    ) -> list[RawIntentHit]:
        query = query.strip()
        if not query or top_k <= 0:
            return []

        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")

        response = await self.es.search(
            index=self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [
                        {"term": {"enabled": True}},
                        {"term": {"review_status": "APPROVED"}},
                    ],
                    "should": [
                        {
                            "match": {
                                "example_text": {
                                    "query": query,
                                    "boost": 2.0,
                                }
                            }
                        },
                        {
                            "match": {
                                "keywords": {
                                    "query": query,
                                    "boost": 1.5,
                                }
                            }
                        },
                        {
                            "match": {
                                "intent_name": {
                                    "query": query,
                                    "boost": 1.0,
                                }
                            }
                        },
                        {
                            "match": {
                                "description": {
                                    "query": query,
                                    "boost": 0.6,
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            source=[
                "intent_code",
                "intent_name",
                "route_target",
                "risk_level",
                "example_text",
                "keywords",
                "embedding",
                "priority",
                "threshold",
                "version",
            ],
        )

        body = response.body if hasattr(response, "body") else response
        raw_hits = body.get("hits", {}).get("hits", [])
        hits: list[RawIntentHit] = []

        for item in raw_hits:
            source = item.get("_source", {})
            intent_code = source.get("intent_code")
            if not intent_code:
                continue

            keywords = source.get("keywords") or []
            embedding = source.get("embedding")

            if not isinstance(keywords, list):
                keywords = []

            if not isinstance(embedding, list):
                embedding = None

            hits.append(
                RawIntentHit(
                    intent_code=intent_code,
                    route_target=source.get("route_target"),
                    risk_level=source.get("risk_level", "LOW"),
                    example_text=source.get("example_text"),
                    keywords=keywords,
                    embedding=embedding,
                    matched_fields=self._infer_matched_fields(
                        query=query,
                        example_text=source.get("example_text"),
                        keywords=keywords,
                    ),
                    bm25_score=float(item.get("_score") or 0.0),
                    metadata={
                        "es_id": item.get("_id"),
                        "index": item.get("_index"),
                        "intent_name": source.get("intent_name"),
                        "priority": source.get("priority"),
                        "threshold": source.get("threshold"),
                        "version": source.get("version"),
                    },
                )
            )

        return hits

    @staticmethod
    def _infer_matched_fields(
        *,
        query: str,
        example_text: str | None,
        keywords: list[str],
    ) -> list[str]:
        matched_fields: list[str] = []
        query_lower = query.lower()

        if example_text:
            matched_fields.append("example_text")

        for keyword in keywords:
            if isinstance(keyword, str) and keyword.lower() in query_lower:
                matched_fields.append("keywords")
                break

        return matched_fields
