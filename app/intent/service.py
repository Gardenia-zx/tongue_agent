from dataclasses import dataclass

from app.core.config import get_settings
from app.integrations.model_gateway import LocalEmbeddingModel
from app.intent.domain_terms import DomainNormalizationResult, DomainTermNormalizer
from app.intent.es_intent_retriever import ESIntentRetriever, RawIntentHit
from app.intent.rules import (
    QueryNormalization,
    RuleIntentMatch,
    match_business_intent,
    match_safety_intent,
    normalize_query_text,
)
from app.intent.similarity import (
    IntentScore,
    calculate_keyword_score,
    clamp_score,
    cosine_similarity,
    fuse_intent_score,
)
from app.schemas.intent import IntentCandidate, IntentDecision, IntentResult, RiskLevel


GENERAL_CHAT_INTENT = "GENERAL_CHAT"
GENERAL_CHAT_ROUTE = "general_chat_subgraph"
HIGH_RISK_ROUTE = "high_risk_safety_subgraph"


@dataclass(frozen=True)
class ScoredHit:
    hit: RawIntentHit
    score: IntentScore


@dataclass(frozen=True)
class AggregatedIntent:
    hit: RawIntentHit
    score: IntentScore
    final_score: float
    hit_count: int
    domain_bonus: float
    coverage_bonus: float


class IntentRecognitionService:
    def __init__(
        self,
        *,
        retriever: ESIntentRetriever,
        embedding_model: LocalEmbeddingModel,
        domain_normalizer: DomainTermNormalizer | None = None,
    ) -> None:
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.domain_normalizer = domain_normalizer
        self.settings = get_settings()

    async def recognize(self, query: str) -> IntentResult:
        raw_query = query.strip()

        if not raw_query:
            return self._general_chat_result(
                detected_intent="UNKNOWN",
                confidence=0.0,
            )

        normalization = normalize_query_text(raw_query)
        normalized_query = normalization.normalized_text

        safety_rule = match_safety_intent(normalized_query)
        if safety_rule is not None:
            return self._rule_result(
                rule=safety_rule,
                normalization=normalization,
                domain_result=None,
            )

        domain_result = await self._normalize_domain_terms(normalized_query)
        domain_risk_result = self._domain_risk_result(
            domain_result=domain_result,
            normalization=normalization,
        )
        if domain_risk_result is not None:
            return domain_risk_result

        business_rule = match_business_intent(normalized_query, domain_result)
        if business_rule is not None:
            return self._rule_result(
                rule=business_rule,
                normalization=normalization,
                domain_result=domain_result,
            )

        search_top_k = min(max(self.settings.intent_top_k * 4, 20), 100)
        hits = await self.retriever.search(
            query=normalized_query,
            top_k=search_top_k,
        )

        if not hits:
            return self._general_chat_result(
                detected_intent="UNKNOWN",
                confidence=0.0,
                normalization=normalization,
                domain_result=domain_result,
            )

        query_embedding = await self.embedding_model.embed_text(normalized_query)
        scored_hits = await self._score_hits(
            hits=hits,
            query=normalized_query,
            query_embedding=query_embedding,
        )
        ranked_candidates = self._aggregate_by_intent(
            scored_hits=scored_hits,
            domain_result=domain_result,
        )
        ranked_candidates = self._remove_es_only_safety_candidates(ranked_candidates)

        if not ranked_candidates:
            return self._general_chat_result(
                detected_intent="UNKNOWN",
                confidence=0.0,
                normalization=normalization,
                domain_result=domain_result,
            )

        best_candidate = ranked_candidates[0]
        best_hit = best_candidate.hit
        decision = self._decide(best_hit, best_candidate.final_score)
        primary_intent = self._primary_intent_for(
            hit=best_hit,
            decision=decision,
            fusion_score=best_candidate.final_score,
        )
        route_target = self._route_target_for(
            hit=best_hit,
            decision=decision,
            fusion_score=best_candidate.final_score,
        )
        safety_flags = self._safety_flags_for(best_hit)

        return IntentResult(
            engine="RULE_ES_BM25_TEXT2VEC_HYBRID",
            engine_version="intent-hybrid-v1.1",
            detected_intent=best_hit.intent_code,
            primary_intent=primary_intent,
            secondary_intents=[
                candidate.hit.intent_code
                for candidate in ranked_candidates[1:]
                if candidate.final_score >= self.settings.intent_clarify_threshold
            ],
            confidence=best_candidate.final_score,
            decision=decision,
            route_target=route_target,
            risk_level=self._risk_level_for(best_hit),
            missing_slots=[],
            entities=self._entities_from_domain_result(domain_result),
            topk_candidates=[
                self._to_candidate(candidate)
                for candidate in ranked_candidates[: self.settings.intent_top_k]
            ],
            safety_flags=safety_flags,
            debug={
                "embedding_model": self.settings.embedding_model_name,
                "original_query": normalization.original_text,
                "normalized_query": normalized_query,
                "retrieval_query": normalized_query,
                "query_replacements": normalization.replacements,
                "normalized_terms": [
                    term.normalized_term for term in domain_result.normalized_terms
                ],
                "scoring": "aggregate_by_intent",
                "bm25_weight": 0.35,
                "vector_weight": 0.5,
                "keyword_weight": 0.15,
            },
        )

    async def _score_hits(
        self,
        *,
        hits: list[RawIntentHit],
        query: str,
        query_embedding: list[float],
    ) -> list[ScoredHit]:
        max_bm25_score = max(hit.bm25_score for hit in hits)

        missing_embedding_hits = [
            hit for hit in hits if hit.embedding is None and hit.example_text
        ]
        fallback_embeddings = await self.embedding_model.embed_texts(
            [hit.example_text or "" for hit in missing_embedding_hits]
        )
        fallback_embedding_map = {
            hit.intent_code + ":" + (hit.example_text or ""): embedding
            for hit, embedding in zip(missing_embedding_hits, fallback_embeddings)
        }

        scored_hits: list[ScoredHit] = []
        for hit in hits:
            candidate_embedding = hit.embedding

            if candidate_embedding is None and hit.example_text:
                candidate_embedding = fallback_embedding_map.get(
                    hit.intent_code + ":" + hit.example_text
                )

            vector_score = (
                cosine_similarity(query_embedding, candidate_embedding)
                if candidate_embedding is not None
                else 0.0
            )
            keyword_score = calculate_keyword_score(query, hit.keywords)
            score = fuse_intent_score(
                bm25_score=hit.bm25_score,
                max_bm25_score=max_bm25_score,
                vector_score=vector_score,
                keyword_score=keyword_score,
            )
            scored_hits.append(ScoredHit(hit=hit, score=score))

        return scored_hits

    def _aggregate_by_intent(
        self,
        *,
        scored_hits: list[ScoredHit],
        domain_result: DomainNormalizationResult,
    ) -> list[AggregatedIntent]:
        grouped: dict[str, list[ScoredHit]] = {}
        for scored_hit in scored_hits:
            grouped.setdefault(scored_hit.hit.intent_code, []).append(scored_hit)

        candidates: list[AggregatedIntent] = []
        for intent_code, intent_hits in grouped.items():
            best_hit = max(intent_hits, key=lambda item: item.score.fusion_score)
            domain_bonus = self._domain_bonus_for_intent(
                intent_code=intent_code,
                domain_result=domain_result,
            )
            coverage_bonus = min(0.05, max(len(intent_hits) - 1, 0) * 0.015)
            final_score = clamp_score(
                best_hit.score.fusion_score + domain_bonus + coverage_bonus
            )
            candidates.append(
                AggregatedIntent(
                    hit=best_hit.hit,
                    score=best_hit.score,
                    final_score=final_score,
                    hit_count=len(intent_hits),
                    domain_bonus=domain_bonus,
                    coverage_bonus=coverage_bonus,
                )
            )

        candidates.sort(key=lambda item: item.final_score, reverse=True)
        return candidates

    @classmethod
    def _remove_es_only_safety_candidates(
        cls,
        candidates: list[AggregatedIntent],
    ) -> list[AggregatedIntent]:
        return [
            candidate
            for candidate in candidates
            if cls._risk_level_for(candidate.hit) not in {RiskLevel.HIGH, RiskLevel.EMERGENCY}
        ]

    @staticmethod
    def _domain_bonus_for_intent(
        *,
        intent_code: str,
        domain_result: DomainNormalizationResult,
    ) -> float:
        related_intents = {
            intent
            for term in domain_result.normalized_terms
            for intent in term.related_intents
        }
        if intent_code in related_intents:
            return 0.08
        return 0.0

    async def _normalize_domain_terms(self, query: str) -> DomainNormalizationResult:
        if self.domain_normalizer is None:
            return DomainNormalizationResult(
                original_query=query,
                normalized_terms=[],
                expanded_query=query,
            )

        return await self.domain_normalizer.normalize(query=query)

    @staticmethod
    def _entities_from_domain_result(
        domain_result: DomainNormalizationResult,
    ):
        return [
            {
                "type": term.term_type,
                "raw_text": term.raw_term,
                "normalized_text": term.normalized_term,
                "confidence": None,
                "source": "domain_term_repository",
                "metadata": {
                    **term.metadata,
                    "term_id": term.term_id,
                    "aliases": term.aliases,
                    "related_intents": term.related_intents,
                    "risk_level": term.risk_level,
                    "score": term.score,
                },
            }
            for term in domain_result.normalized_terms
        ]

    def _domain_risk_result(
        self,
        *,
        domain_result: DomainNormalizationResult,
        normalization: QueryNormalization,
    ) -> IntentResult | None:
        if not domain_result.normalized_terms:
            return None

        direct_risk_terms = [
            term
            for term in domain_result.normalized_terms
            if self._is_direct_risk_match(
                query=domain_result.original_query,
                raw_term=term.raw_term,
                aliases=term.aliases,
                risk_level=term.risk_level,
            )
        ]

        if not direct_risk_terms:
            return None

        risk_order = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.EMERGENCY: 3,
        }
        highest_risk = RiskLevel.LOW
        related_intents: list[str] = []

        for term in direct_risk_terms:
            try:
                term_risk = RiskLevel(term.risk_level)
            except ValueError:
                term_risk = RiskLevel.LOW

            if risk_order[term_risk] > risk_order[highest_risk]:
                highest_risk = term_risk

            related_intents.extend(term.related_intents)

        if highest_risk == RiskLevel.EMERGENCY:
            detected_intent = (
                "EMERGENCY_MEDICAL"
                if "EMERGENCY_MEDICAL" in related_intents
                else "HIGH_RISK_MEDICAL"
            )
            return IntentResult(
                engine="RULE_ES_BM25_TEXT2VEC_HYBRID",
                engine_version="intent-hybrid-v1.1",
                detected_intent=detected_intent,
                primary_intent=detected_intent,
                secondary_intents=[],
                confidence=1.0,
                decision=IntentDecision.ROUTE,
                route_target=HIGH_RISK_ROUTE,
                risk_level=RiskLevel.EMERGENCY,
                entities=self._entities_from_domain_terms(direct_risk_terms),
                safety_flags=["EMERGENCY_INTENT"],
                debug={
                    "risk_override": "domain_term_repository",
                    "original_query": normalization.original_text,
                    "normalized_query": normalization.normalized_text,
                    "query_replacements": normalization.replacements,
                    "normalized_terms": [
                        term.normalized_term for term in direct_risk_terms
                    ],
                },
            )

        if highest_risk == RiskLevel.HIGH:
            return IntentResult(
                engine="RULE_ES_BM25_TEXT2VEC_HYBRID",
                engine_version="intent-hybrid-v1.1",
                detected_intent="HIGH_RISK_MEDICAL",
                primary_intent="HIGH_RISK_MEDICAL",
                secondary_intents=[],
                confidence=1.0,
                decision=IntentDecision.ROUTE,
                route_target=HIGH_RISK_ROUTE,
                risk_level=RiskLevel.HIGH,
                entities=self._entities_from_domain_terms(direct_risk_terms),
                safety_flags=["HIGH_RISK_INTENT"],
                debug={
                    "risk_override": "domain_term_repository",
                    "original_query": normalization.original_text,
                    "normalized_query": normalization.normalized_text,
                    "query_replacements": normalization.replacements,
                    "normalized_terms": [
                        term.normalized_term for term in direct_risk_terms
                    ],
                },
            )

        return None

    @staticmethod
    def _is_direct_risk_match(
        *,
        query: str,
        raw_term: str,
        aliases: list[str],
        risk_level: str,
    ) -> bool:
        try:
            parsed_risk = RiskLevel(risk_level)
        except ValueError:
            return False

        if parsed_risk not in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
            return False

        terms = [raw_term, *aliases]
        return any(term and term in query for term in terms)

    @staticmethod
    def _entities_from_domain_terms(terms):
        return [
            {
                "type": term.term_type,
                "raw_text": term.raw_term,
                "normalized_text": term.normalized_term,
                "confidence": None,
                "source": "domain_term_repository",
                "metadata": {
                    **term.metadata,
                    "term_id": term.term_id,
                    "aliases": term.aliases,
                    "related_intents": term.related_intents,
                    "risk_level": term.risk_level,
                    "score": term.score,
                },
            }
            for term in terms
        ]

    def _decide(
        self,
        hit: RawIntentHit,
        fusion_score: float,
    ) -> IntentDecision:
        risk_level = self._risk_level_for(hit)

        if risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
            return IntentDecision.ROUTE

        if hit.intent_code == GENERAL_CHAT_INTENT:
            return IntentDecision.ROUTE

        if fusion_score >= self._route_threshold_for(hit):
            return IntentDecision.ROUTE

        if fusion_score >= self._clarify_threshold_for(hit):
            return IntentDecision.CLARIFY

        return IntentDecision.ROUTE

    def _primary_intent_for(
        self,
        *,
        hit: RawIntentHit,
        decision: IntentDecision,
        fusion_score: float,
    ) -> str:
        if self._risk_level_for(hit) in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
            return hit.intent_code

        if hit.intent_code == GENERAL_CHAT_INTENT:
            return GENERAL_CHAT_INTENT

        if decision == IntentDecision.CLARIFY:
            return GENERAL_CHAT_INTENT

        if fusion_score < self._clarify_threshold_for(hit):
            return GENERAL_CHAT_INTENT

        return hit.intent_code

    def _route_target_for(
        self,
        *,
        hit: RawIntentHit,
        decision: IntentDecision,
        fusion_score: float,
    ) -> str:
        if self._risk_level_for(hit) in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
            return hit.route_target or HIGH_RISK_ROUTE

        if hit.intent_code == GENERAL_CHAT_INTENT:
            return GENERAL_CHAT_ROUTE

        if decision == IntentDecision.CLARIFY:
            return GENERAL_CHAT_ROUTE

        if fusion_score < self._clarify_threshold_for(hit):
            return GENERAL_CHAT_ROUTE

        return hit.route_target or GENERAL_CHAT_ROUTE

    @staticmethod
    def _risk_level_for(hit: RawIntentHit) -> RiskLevel:
        try:
            return RiskLevel(hit.risk_level)
        except ValueError:
            return RiskLevel.LOW

    def _route_threshold_for(self, hit: RawIntentHit) -> float:
        threshold = hit.metadata.get("threshold")
        if isinstance(threshold, dict):
            route_threshold = threshold.get("route")
            if isinstance(route_threshold, (int, float)):
                return float(route_threshold)
        return self.settings.intent_route_threshold

    def _clarify_threshold_for(self, hit: RawIntentHit) -> float:
        threshold = hit.metadata.get("threshold")
        if isinstance(threshold, dict):
            clarify_threshold = threshold.get("clarify")
            if isinstance(clarify_threshold, (int, float)):
                return float(clarify_threshold)
        return self.settings.intent_clarify_threshold

    @classmethod
    def _safety_flags_for(cls, hit: RawIntentHit) -> list[str]:
        risk_level = cls._risk_level_for(hit)

        if risk_level == RiskLevel.EMERGENCY:
            return ["EMERGENCY_INTENT"]

        if risk_level == RiskLevel.HIGH:
            return ["HIGH_RISK_INTENT"]

        return []

    @staticmethod
    def _to_candidate(candidate: AggregatedIntent) -> IntentCandidate:
        return IntentCandidate(
            intent=candidate.hit.intent_code,
            score=candidate.final_score,
            matched_text=candidate.hit.example_text,
            matched_fields=candidate.hit.matched_fields,
            route_target=candidate.hit.route_target,
            risk_level=IntentRecognitionService._risk_level_for(candidate.hit),
            metadata={
                **candidate.hit.metadata,
                "bm25_score": candidate.score.bm25_score,
                "vector_score": candidate.score.vector_score,
                "keyword_score": candidate.score.keyword_score,
                "base_fusion_score": candidate.score.fusion_score,
                "domain_bonus": candidate.domain_bonus,
                "coverage_bonus": candidate.coverage_bonus,
                "aggregate_hit_count": candidate.hit_count,
            },
        )

    @staticmethod
    def _rule_result(
        *,
        rule: RuleIntentMatch,
        normalization: QueryNormalization,
        domain_result: DomainNormalizationResult | None,
    ) -> IntentResult:
        normalized_terms = []
        entities = []
        if domain_result is not None:
            normalized_terms = [
                term.normalized_term for term in domain_result.normalized_terms
            ]
            entities = IntentRecognitionService._entities_from_domain_result(
                domain_result
            )

        return IntentResult(
            engine="RULE_ES_BM25_TEXT2VEC_HYBRID",
            engine_version="intent-hybrid-v1.1",
            detected_intent=rule.intent_code,
            primary_intent=rule.intent_code,
            secondary_intents=[],
            confidence=rule.confidence,
            decision=rule.decision,
            route_target=rule.route_target,
            risk_level=rule.risk_level,
            missing_slots=[],
            entities=entities,
            topk_candidates=[
                IntentCandidate(
                    intent=rule.intent_code,
                    score=rule.confidence,
                    matched_text=normalization.normalized_text,
                    matched_fields=["hard_rule"],
                    route_target=rule.route_target,
                    risk_level=rule.risk_level,
                    metadata={
                        "matched_rule": rule.matched_rule,
                        "matched_texts": rule.matched_texts,
                    },
                )
            ],
            safety_flags=rule.safety_flags,
            debug={
                "rule_match": rule.matched_rule,
                "original_query": normalization.original_text,
                "normalized_query": normalization.normalized_text,
                "query_replacements": normalization.replacements,
                "normalized_terms": normalized_terms,
            },
        )

    @staticmethod
    def _general_chat_result(
        *,
        detected_intent: str,
        confidence: float,
        normalization: QueryNormalization | None = None,
        domain_result: DomainNormalizationResult | None = None,
    ) -> IntentResult:
        debug = {}
        if normalization is not None:
            debug = {
                "original_query": normalization.original_text,
                "normalized_query": normalization.normalized_text,
                "query_replacements": normalization.replacements,
            }
        if domain_result is not None:
            debug["normalized_terms"] = [
                term.normalized_term for term in domain_result.normalized_terms
            ]

        return IntentResult(
            engine="RULE_ES_BM25_TEXT2VEC_HYBRID",
            engine_version="intent-hybrid-v1.1",
            detected_intent=detected_intent,
            primary_intent=GENERAL_CHAT_INTENT,
            secondary_intents=[],
            confidence=confidence,
            decision=IntentDecision.ROUTE,
            route_target=GENERAL_CHAT_ROUTE,
            risk_level=RiskLevel.LOW,
            debug=debug,
        )
