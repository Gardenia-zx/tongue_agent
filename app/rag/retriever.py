from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.intent.similarity import clamp_score
from app.rag.query_expansion import expand_rag_query
from app.schemas.rag import RagRetrievalHit


@dataclass
class _PartialHit:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    source_uri: str | None
    tags: list[str]
    metadata: dict[str, Any]
    bm25_score: float = 0.0
    vector_score: float = 0.0
    lexical_score: float = 0.0


class HybridRagRetriever:
    def __init__(
        self,
        es: AsyncElasticsearch,
        *,
        index_name: str,
    ) -> None:
        self.es = es
        self.index_name = index_name

    async def retrieve(
        self,
        *,
        query: str,
        query_embedding: list[float],
        bm25_top_k: int,
        vector_top_k: int,
        final_top_k: int,
        min_relevance_score: float,
    ) -> list[RagRetrievalHit]:
        query = query.strip()
        if not query:
            return []

        expanded_query = expand_rag_query(query)
        bm25_hits = await self._bm25_search(
            query=expanded_query.search_text,
            raw_query=expanded_query.raw_query,
            expanded_terms=expanded_query.terms,
            top_k=bm25_top_k,
        )
        vector_hits = await self._vector_search(
            query_embedding=query_embedding,
            top_k=vector_top_k,
        )

        merged: dict[str, _PartialHit] = {}
        max_bm25 = max([hit.bm25_score for hit in bm25_hits], default=0.0)

        for hit in bm25_hits:
            normalized_bm25 = hit.bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            existing = merged.get(hit.chunk_id)
            if existing is None:
                hit.bm25_score = clamp_score(normalized_bm25)
                merged[hit.chunk_id] = hit
            else:
                existing.bm25_score = max(
                    existing.bm25_score,
                    clamp_score(normalized_bm25),
                )

        for hit in vector_hits:
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = hit
            else:
                existing.vector_score = max(existing.vector_score, hit.vector_score)

        ranked_hits: list[RagRetrievalHit] = []
        for hit in merged.values():
            hit.lexical_score = self._lexical_score(
                query=query,
                expanded_terms=expanded_query.terms,
                hit=hit,
            )
            final_score = clamp_score(
                hit.bm25_score * 0.25
                + hit.vector_score * 0.35
                + hit.lexical_score * 0.40
            )
            if final_score < min_relevance_score:
                continue
            ranked_hits.append(
                RagRetrievalHit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    title=hit.title,
                    content=hit.content,
                    source_uri=hit.source_uri,
                    tags=hit.tags,
                    bm25_score=hit.bm25_score,
                    vector_score=hit.vector_score,
                    final_score=final_score,
                    metadata={
                        **hit.metadata,
                        "lexical_score": hit.lexical_score,
                    },
                )
            )

        ranked_hits.sort(key=lambda item: item.final_score, reverse=True)
        return ranked_hits[:final_top_k]

    async def _bm25_search(
        self,
        *,
        query: str,
        raw_query: str,
        expanded_terms: list[str],
        top_k: int,
    ) -> list[_PartialHit]:
        should_clauses: list[dict[str, Any]] = [
            {
                "match": {
                    "content": {
                        "query": query,
                        "boost": 2.0,
                    }
                }
            },
            {
                "match": {
                    "title": {
                        "query": query,
                        "boost": 1.2,
                    }
                }
            },
            {
                "terms": {
                    "tags": [raw_query],
                    "boost": 0.8,
                }
            },
        ]

        phrase_terms = [raw_query, *expanded_terms]
        for term in phrase_terms[:24]:
            if not term:
                continue
            should_clauses.extend(
                [
                    {
                        "match_phrase": {
                            "content": {
                                "query": term,
                                "boost": 3.0,
                            }
                        }
                    },
                    {
                        "wildcard": {
                            "content.raw": {
                                "value": f"*{term}*",
                                "boost": 2.5,
                            }
                        }
                    },
                ]
            )

        response = await self.es.search(
            index=self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [{"term": {"enabled": True}}],
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
            source=[
                "chunk_id",
                "doc_id",
                "title",
                "content",
                "source_uri",
                "tags",
                "metadata",
            ],
        )
        hits = self._parse_hits(response)
        for hit in hits:
            hit.vector_score = 0.0
        return hits

    async def _vector_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
    ) -> list[_PartialHit]:
        if not query_embedding:
            return []

        response = await self.es.search(
            index=self.index_name,
            size=top_k,
            query={
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [{"term": {"enabled": True}}],
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding},
                    },
                }
            },
            source=[
                "chunk_id",
                "doc_id",
                "title",
                "content",
                "source_uri",
                "tags",
                "metadata",
            ],
        )
        hits = self._parse_hits(response)
        for hit in hits:
            hit.bm25_score = 0.0
            hit.vector_score = clamp_score(hit.vector_score)
        return hits

    @staticmethod
    def _parse_hits(response: Any) -> list[_PartialHit]:
        body = response.body if hasattr(response, "body") else response
        raw_hits = body.get("hits", {}).get("hits", [])
        hits: list[_PartialHit] = []

        for item in raw_hits:
            source = item.get("_source", {})
            tags = source.get("tags") or []
            metadata = source.get("metadata") or {}
            if not isinstance(tags, list):
                tags = []
            if not isinstance(metadata, dict):
                metadata = {}

            raw_score = float(item.get("_score") or 0.0)
            vector_score = clamp_score(raw_score - 1.0)
            hits.append(
                _PartialHit(
                    chunk_id=source.get("chunk_id", item.get("_id")),
                    doc_id=source.get("doc_id", ""),
                    title=source.get("title", ""),
                    content=source.get("content", ""),
                    source_uri=source.get("source_uri"),
                    tags=tags,
                    metadata=metadata,
                    bm25_score=raw_score,
                    vector_score=vector_score,
                )
            )

        return hits

    @staticmethod
    def _lexical_score(
        *,
        query: str,
        expanded_terms: list[str],
        hit: _PartialHit,
    ) -> float:
        compact_query = "".join(query.split())
        if not compact_query:
            return 0.0

        tag_score = 0.0
        for tag in hit.tags:
            if tag and (tag in compact_query or compact_query in tag):
                tag_score = max(tag_score, 1.0)

        searchable_text = f"{hit.title}\n{hit.content}"
        if compact_query in searchable_text:
            return 1.0

        term_score = 0.0
        if expanded_terms:
            matched_terms = [term for term in expanded_terms if term and term in searchable_text]
            if matched_terms:
                term_score = min(1.0, 0.35 + len(matched_terms) / 4)

        grams = {
            compact_query[index : index + 2]
            for index in range(max(len(compact_query) - 1, 0))
            if len(compact_query[index : index + 2]) == 2
        }
        if not grams:
            return clamp_score(max(tag_score, term_score))

        matched = sum(1 for gram in grams if gram in searchable_text)
        gram_score = min(1.0, matched / max(len(grams), 1) * 1.4)
        return clamp_score(max(tag_score, term_score, gram_score))
