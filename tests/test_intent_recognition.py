import os
import unittest
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.core.config import get_settings
from app.integrations.es_client import create_es_client
from app.integrations.model_gateway import get_embedding_model
from app.intent.domain_terms import DomainTermNormalizer
from app.intent.es_intent_retriever import ESIntentRetriever
from app.intent.es_intent_retriever import RawIntentHit
from app.intent.rules import match_safety_intent
from app.intent.service import AggregatedIntent, IntentRecognitionService
from app.intent.similarity import IntentScore
from app.schemas.intent import IntentDecision, RiskLevel


@dataclass(frozen=True)
class IntentCase:
    text: str
    expected_intent: str
    expected_route: str
    expected_risk: RiskLevel = RiskLevel.LOW
    min_confidence: float = 0.5


class TestIntentRecognition(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    async def test_recognize_main_and_fuzzy_intents(self) -> None:
        cases = [
            IntentCase("你好，你能做什么", "GENERAL_CHAT", "general_chat_subgraph"),
            IntentCase("这个系统怎么用", "GENERAL_CHAT", "general_chat_subgraph"),
            IntentCase("你可以帮我什么", "GENERAL_CHAT", "general_chat_subgraph"),
            IntentCase("我想先了解一下功能", "GENERAL_CHAT", "general_chat_subgraph"),
            IntentCase(
                "我想做一次舌像分析",
                "TONGUE_ANALYSIS_START",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "上传舌头照片帮我分析一下",
                "TONGUE_ANALYSIS_START",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "帮我看看舌头",
                "TONGUE_ANALYSIS_START",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "我拍了舌头照片，帮我看看",
                "TONGUE_ANALYSIS_START",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "继续刚才的分析",
                "TONGUE_ANALYSIS_CONTINUE",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "我来回答刚才的问题",
                "TONGUE_ANALYSIS_CONTINUE",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "我补充一下症状",
                "TONGUE_ANALYSIS_CONTINUE",
                "tongue_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "告诉我湿气重是什么意思",
                "HEALTH_KNOWLEDGE_QA",
                "health_qa_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "舌头有齿痕代表什么",
                "HEALTH_KNOWLEDGE_QA",
                "health_qa_subgraph",
                min_confidence=0.9,
            ),
            IntentCase("白苔和黄苔有什么区别", "HEALTH_KNOWLEDGE_QA", "health_qa_subgraph"),
            IntentCase("痰湿体质平时怎么调理", "HEALTH_KNOWLEDGE_QA", "health_qa_subgraph"),
            IntentCase(
                "解释一下我的报告",
                "REPORT_EXPLANATION",
                "report_explanation_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "报告里说痰湿是什么意思",
                "REPORT_EXPLANATION",
                "report_explanation_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "帮我解释这份舌象报告",
                "REPORT_EXPLANATION",
                "report_explanation_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "讲详细一点",
                "REPORT_EXPLANATION",
                "general_chat_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "对我的舌象分析部分，分析的详细一点",
                "REPORT_EXPLANATION",
                "general_chat_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "看看我最近几次报告有没有变化",
                "TREND_ANALYSIS",
                "trend_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "和上次相比怎么样",
                "TREND_ANALYSIS",
                "trend_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "帮我分析一下历史趋势",
                "TREND_ANALYSIS",
                "trend_analysis_subgraph",
                min_confidence=0.9,
            ),
            IntentCase(
                "删除我的报告",
                "PRIVACY_REQUEST",
                "privacy_request_subgraph",
                RiskLevel.MEDIUM,
                min_confidence=0.9,
            ),
            IntentCase(
                "把这张舌头照片删掉",
                "PRIVACY_REQUEST",
                "privacy_request_subgraph",
                RiskLevel.MEDIUM,
                min_confidence=0.9,
            ),
            IntentCase(
                "撤回长期记忆授权",
                "PRIVACY_REQUEST",
                "privacy_request_subgraph",
                RiskLevel.MEDIUM,
                min_confidence=0.9,
            ),
            IntentCase(
                "我应该吃什么药",
                "HIGH_RISK_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.HIGH,
                min_confidence=1.0,
            ),
            IntentCase(
                "这个情况要不要停药",
                "HIGH_RISK_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.HIGH,
                min_confidence=1.0,
            ),
            IntentCase(
                "给我一个治疗方案",
                "HIGH_RISK_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.HIGH,
                min_confidence=1.0,
            ),
            IntentCase(
                "我现在胸好痛",
                "EMERGENCY_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.EMERGENCY,
                min_confidence=1.0,
            ),
            IntentCase(
                "喘不上气怎么办",
                "EMERGENCY_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.EMERGENCY,
                min_confidence=1.0,
            ),
            IntentCase(
                "严重过敏喘不上气",
                "EMERGENCY_MEDICAL",
                "high_risk_safety_subgraph",
                RiskLevel.EMERGENCY,
                min_confidence=1.0,
            ),
        ]

        settings = get_settings()
        es = create_es_client()
        try:
            try:
                await es.info()
            except Exception as exc:
                self.fail(
                    "Elasticsearch is not reachable. Start Docker services and "
                    "run `python -m app.intent.indexer` before this test. "
                    f"Original error: {exc}"
                )

            service = IntentRecognitionService(
                retriever=ESIntentRetriever(es, index_name=settings.intent_index_name),
                embedding_model=get_embedding_model(),
                domain_normalizer=DomainTermNormalizer(
                    es,
                    index_name=settings.domain_term_index_name,
                ),
            )

            for case in cases:
                with self.subTest(text=case.text):
                    result = await service.recognize(case.text)

                    self.assertEqual(case.expected_intent, result.primary_intent)
                    self.assertEqual(case.expected_intent, result.detected_intent)
                    self.assertEqual(IntentDecision.ROUTE, result.decision)
                    self.assertEqual(case.expected_route, result.route_target)
                    self.assertEqual(case.expected_risk, result.risk_level)
                    self.assertGreaterEqual(result.confidence, case.min_confidence)
        finally:
            await es.close()


class TestIntentSafetyRules(unittest.TestCase):
    def test_diet_recommendation_is_not_high_risk_safety(self) -> None:
        self.assertIsNone(match_safety_intent("有什么饮食方面的推荐吗"))
        self.assertIsNone(match_safety_intent("结合我的舌象报告，饮食上怎么注意"))

    def test_explicit_medication_and_diagnosis_are_high_risk(self) -> None:
        for text in [
            "这个药一天吃几片",
            "我应该吃什么中成药",
            "帮我判断是不是某某病",
        ]:
            with self.subTest(text=text):
                match = match_safety_intent(text)
                self.assertIsNotNone(match)
                self.assertEqual("HIGH_RISK_MEDICAL", match.intent_code)
                self.assertEqual(RiskLevel.HIGH, match.risk_level)

    def test_es_only_high_risk_candidates_are_filtered(self) -> None:
        high_risk_hit = RawIntentHit(
            intent_code="HIGH_RISK_MEDICAL",
            route_target="high_risk_safety_subgraph",
            risk_level="HIGH",
            example_text="给我一个治疗方案",
            keywords=["治疗方案"],
            embedding=None,
            matched_fields=["example_text"],
            bm25_score=10.0,
            metadata={},
        )
        health_hit = RawIntentHit(
            intent_code="HEALTH_KNOWLEDGE_QA",
            route_target="health_qa_subgraph",
            risk_level="LOW",
            example_text="饮食怎么注意",
            keywords=["饮食"],
            embedding=None,
            matched_fields=["example_text"],
            bm25_score=3.0,
            metadata={},
        )
        score = IntentScore(
            bm25_score=1.0,
            vector_score=0.8,
            keyword_score=0.4,
            fusion_score=0.9,
        )
        filtered = IntentRecognitionService._remove_es_only_safety_candidates(
            [
                AggregatedIntent(high_risk_hit, score, 0.95, 1, 0.0, 0.0),
                AggregatedIntent(health_hit, score, 0.65, 1, 0.0, 0.0),
            ]
        )

        self.assertEqual(["HEALTH_KNOWLEDGE_QA"], [item.hit.intent_code for item in filtered])
