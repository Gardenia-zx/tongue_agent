from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch


@dataclass(frozen=True)
class DomainTermHit:
    term_id: str
    raw_term: str
    normalized_term: str
    term_type: str
    aliases: list[str]
    related_intents: list[str]
    risk_level: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DomainNormalizationResult:
    original_query: str
    normalized_terms: list[DomainTermHit]
    expanded_query: str


class DomainTermNormalizer:
    def __init__(
        self,
        es: AsyncElasticsearch,
        *,
        index_name: str,
    ) -> None:
        self.es = es
        self.index_name = index_name

    async def normalize(
        self,
        *,
        query: str,
        top_k: int = 1000,
    ) -> DomainNormalizationResult:
        query = query.strip()
        if not query:
            return DomainNormalizationResult(
                original_query=query,
                normalized_terms=[],
                expanded_query=query,
            )

        response = await self.es.search(
            index=self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [
                        {"term": {"enabled": True}},
                    ]
                }
            },
            source=[
                "term_id",
                "raw_term",
                "normalized_term",
                "term_type",
                "aliases",
                "related_intents",
                "risk_level",
                "version",
            ],
        )

        body = response.body if hasattr(response, "body") else response
        raw_hits = body.get("hits", {}).get("hits", [])
        term_hits: list[DomainTermHit] = []

        for item in raw_hits:
            source = item.get("_source", {})
            aliases = source.get("aliases") or []
            related_intents = source.get("related_intents") or []

            if not isinstance(aliases, list):
                aliases = []
            if not isinstance(related_intents, list):
                related_intents = []

            raw_term = source.get("raw_term", "")
            matched_terms = self._matched_terms(
                query=query,
                raw_term=raw_term,
                aliases=aliases,
            )
            if not matched_terms:
                continue

            term_hits.append(
                DomainTermHit(
                    term_id=source["term_id"],
                    raw_term=raw_term,
                    normalized_term=source.get("normalized_term", ""),
                    term_type=source.get("term_type", ""),
                    aliases=aliases,
                    related_intents=related_intents,
                    risk_level=source.get("risk_level", "LOW"),
                    score=1.0,
                    metadata={
                        "es_id": item.get("_id"),
                        "index": item.get("_index"),
                        "version": source.get("version"),
                        "matched_terms": matched_terms,
                    },
                )
            )

        return DomainNormalizationResult(
            original_query=query,
            normalized_terms=term_hits,
            expanded_query=query,
        )

    @staticmethod
    def _matched_terms(
        *,
        query: str,
        raw_term: str,
        aliases: list[str],
    ) -> list[str]:
        terms = [raw_term, *aliases]
        return [term for term in terms if term and term in query]
