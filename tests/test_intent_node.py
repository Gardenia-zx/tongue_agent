import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.agent.nodes.intent_node import intent_node


class _FakeES:
    async def close(self) -> None:
        return None


class _FakeIntentResult:
    def __init__(
        self,
        *,
        detected_intent: str = "GENERAL_CHAT",
        primary_intent: str = "GENERAL_CHAT",
        route_target: str = "general_chat_subgraph",
        risk_level: str = "LOW",
        safety_flags: list[str] | None = None,
    ) -> None:
        self.detected_intent = detected_intent
        self.primary_intent = primary_intent
        self.route_target = route_target
        self.risk_level = risk_level
        self.safety_flags = safety_flags or []

    def model_dump(self, mode: str = "json") -> dict:
        return {
            "schema_version": "1.0",
            "engine": "TEST",
            "engine_version": "test",
            "detected_intent": self.detected_intent,
            "primary_intent": self.primary_intent,
            "secondary_intents": [],
            "confidence": 0.72,
            "decision": "ROUTE",
            "route_target": self.route_target,
            "risk_level": self.risk_level,
            "missing_slots": [],
            "entities": [],
            "topk_candidates": [],
            "safety_flags": self.safety_flags,
            "debug": {},
        }


class TestIntentNode(unittest.IsolatedAsyncioTestCase):
    async def test_recognize_receives_only_standalone_query_then_applies_route_hint(self) -> None:
        service = Mock()
        service.recognize = AsyncMock(return_value=_FakeIntentResult())
        state = {
            "turn_id": "turn-1",
            "message": {"content": "第二个详细点"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "standalone_query": "请解释上一轮回答中的第二项：睡眠不好。",
                    "reference_resolution": {"target_type": "LAST_ANSWER_ITEM"},
                    "route_hint": "health_qa_subgraph",
                    "clarification_status": "NOT_NEEDED",
                    "prompt_context": {
                        "recent_messages": [{"content": "old context that must not be searched"}],
                        "conversation_summary": "summary that must not be searched",
                    },
                },
            },
        }

        with (
            patch("app.agent.nodes.intent_node.create_es_client", return_value=_FakeES()),
            patch("app.agent.nodes.intent_node.ESIntentRetriever", return_value=Mock()),
            patch("app.agent.nodes.intent_node.DomainTermNormalizer", return_value=Mock()),
            patch("app.agent.nodes.intent_node._get_embedding_model", return_value=Mock()),
            patch("app.agent.nodes.intent_node._build_intent_service", return_value=service),
        ):
            result = await intent_node(state)

        service.recognize.assert_awaited_once_with("请解释上一轮回答中的第二项：睡眠不好。")
        self.assertEqual("health_qa_subgraph", result["intent_result"]["route_target"])
        self.assertEqual("health_qa_subgraph", result["next_action"]["payload"]["route_target"])
        self.assertEqual(
            "LAST_ANSWER_ITEM",
            result["intent_result"]["debug"]["query_rewrite_reference"]["target_type"],
        )

    async def test_report_diet_route_hint_overrides_high_risk_false_positive(self) -> None:
        service = Mock()
        service.recognize = AsyncMock(
            return_value=_FakeIntentResult(
                detected_intent="HIGH_RISK_MEDICAL",
                primary_intent="HIGH_RISK_MEDICAL",
                route_target="high_risk_safety_subgraph",
                risk_level="HIGH",
                safety_flags=["HIGH_RISK_INTENT"],
            )
        )
        state = {
            "turn_id": "turn-1",
            "message": {"content": "帮我规划一下饮食"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_rewrite_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "帮我规划一下饮食",
                    "active_report_ref": {
                        "report_id": 22,
                        "trusted": True,
                        "feature_summary": "白苔",
                    },
                },
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "帮我规划一下饮食",
                    "standalone_query": "请结合当前活动舌象报告，规划饮食建议。",
                    "reference_resolution": {
                        "target_type": "ACTIVE_REPORT",
                        "target_focus": "DIET_ADVICE",
                    },
                    "route_hint": "report_followup_subgraph",
                    "clarification_status": "NOT_NEEDED",
                },
            },
        }

        with (
            patch("app.agent.nodes.intent_node.create_es_client", return_value=_FakeES()),
            patch("app.agent.nodes.intent_node.ESIntentRetriever", return_value=Mock()),
            patch("app.agent.nodes.intent_node.DomainTermNormalizer", return_value=Mock()),
            patch("app.agent.nodes.intent_node._get_embedding_model", return_value=Mock()),
            patch("app.agent.nodes.intent_node._build_intent_service", return_value=service),
        ):
            result = await intent_node(state)

        self.assertEqual("report_followup_subgraph", result["intent_result"]["route_target"])
        self.assertEqual("LOW", result["intent_result"]["risk_level"])
        self.assertEqual([], result["intent_result"]["safety_flags"])

    async def test_report_load_plan_overrides_high_risk_false_positive(self) -> None:
        service = Mock()
        service.recognize = AsyncMock(
            return_value=_FakeIntentResult(
                detected_intent="HIGH_RISK_MEDICAL",
                primary_intent="HIGH_RISK_MEDICAL",
                route_target="high_risk_safety_subgraph",
                risk_level="HIGH",
                safety_flags=["HIGH_RISK_INTENT"],
            )
        )
        state = {
            "turn_id": "turn-1",
            "message": {"content": "帮我规划一下饮食习惯改善一下"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "帮我规划一下饮食习惯改善一下",
                    "standalone_query": "请结合当前活动舌象报告规划饮食习惯改善建议。",
                    "reference_resolution": {
                        "target_type": "ACTIVE_REPORT",
                        "target_focus": "DIET_ADVICE",
                    },
                    "report_load_plan": {
                        "need_report": True,
                        "target_report_id": 22,
                        "target_type": "ACTIVE_REPORT",
                        "target_focus": "DIET_ADVICE",
                        "sections": ["feature_summary", "interpretation", "dietary_advice"],
                    },
                    "route_hint": "report_followup_subgraph",
                    "clarification_status": "NOT_NEEDED",
                },
            },
        }

        with (
            patch("app.agent.nodes.intent_node.create_es_client", return_value=_FakeES()),
            patch("app.agent.nodes.intent_node.ESIntentRetriever", return_value=Mock()),
            patch("app.agent.nodes.intent_node.DomainTermNormalizer", return_value=Mock()),
            patch("app.agent.nodes.intent_node._get_embedding_model", return_value=Mock()),
            patch("app.agent.nodes.intent_node._build_intent_service", return_value=service),
        ):
            result = await intent_node(state)

        self.assertEqual("report_followup_subgraph", result["intent_result"]["route_target"])
        self.assertEqual("LOW", result["intent_result"]["risk_level"])
        self.assertEqual([], result["intent_result"]["safety_flags"])

    async def test_safe_diet_request_without_report_does_not_enter_safety(self) -> None:
        service = Mock()
        service.recognize = AsyncMock(
            return_value=_FakeIntentResult(
                detected_intent="HIGH_RISK_MEDICAL",
                primary_intent="HIGH_RISK_MEDICAL",
                route_target="high_risk_safety_subgraph",
                risk_level="HIGH",
                safety_flags=["HIGH_RISK_INTENT"],
            )
        )
        state = {
            "turn_id": "turn-1",
            "message": {"content": "帮我规划一下饮食习惯改善一下"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "帮我规划一下饮食习惯改善一下",
                    "standalone_query": "帮我规划一下饮食习惯改善一下",
                    "reference_resolution": {"target_type": "GENERAL_TOPIC"},
                    "clarification_status": "NOT_NEEDED",
                },
            },
        }

        with (
            patch("app.agent.nodes.intent_node.create_es_client", return_value=_FakeES()),
            patch("app.agent.nodes.intent_node.ESIntentRetriever", return_value=Mock()),
            patch("app.agent.nodes.intent_node.DomainTermNormalizer", return_value=Mock()),
            patch("app.agent.nodes.intent_node._get_embedding_model", return_value=Mock()),
            patch("app.agent.nodes.intent_node._build_intent_service", return_value=service),
        ):
            result = await intent_node(state)

        self.assertEqual("health_qa_subgraph", result["intent_result"]["route_target"])
        self.assertEqual("LOW", result["intent_result"]["risk_level"])
        self.assertEqual([], result["intent_result"]["safety_flags"])

    async def test_report_route_hint_keeps_explicit_diagnosis_in_safety(self) -> None:
        service = Mock()
        service.recognize = AsyncMock(
            return_value=_FakeIntentResult(
                detected_intent="HIGH_RISK_MEDICAL",
                primary_intent="HIGH_RISK_MEDICAL",
                route_target="high_risk_safety_subgraph",
                risk_level="HIGH",
                safety_flags=["HIGH_RISK_INTENT"],
            )
        )
        state = {
            "turn_id": "turn-1",
            "message": {"content": "结合报告判断我是不是肾病，饮食怎么吃"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_rewrite_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "结合报告判断我是不是肾病，饮食怎么吃",
                    "active_report_ref": {
                        "report_id": 22,
                        "trusted": True,
                        "feature_summary": "白苔",
                    },
                },
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "raw_user_input": "结合报告判断我是不是肾病，饮食怎么吃",
                    "standalone_query": "请结合当前活动舌象报告判断我是不是肾病，并说明饮食怎么吃。",
                    "reference_resolution": {
                        "target_type": "ACTIVE_REPORT",
                        "target_focus": "DIET_ADVICE",
                    },
                    "route_hint": "report_followup_subgraph",
                    "clarification_status": "NOT_NEEDED",
                },
            },
        }

        with (
            patch("app.agent.nodes.intent_node.create_es_client", return_value=_FakeES()),
            patch("app.agent.nodes.intent_node.ESIntentRetriever", return_value=Mock()),
            patch("app.agent.nodes.intent_node.DomainTermNormalizer", return_value=Mock()),
            patch("app.agent.nodes.intent_node._get_embedding_model", return_value=Mock()),
            patch("app.agent.nodes.intent_node._build_intent_service", return_value=service),
        ):
            result = await intent_node(state)

        self.assertEqual("high_risk_safety_subgraph", result["intent_result"]["route_target"])
        self.assertEqual("HIGH", result["intent_result"]["risk_level"])

    async def test_clarification_status_skips_intent_retrieval(self) -> None:
        state = {
            "turn_id": "turn-1",
            "message": {"content": "继续说"},
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {
                    "turn_id": "turn-1",
                    "context_version": "query_rewrite_context.v1",
                    "standalone_query": "继续说",
                    "reference_resolution": {"target_type": "AMBIGUOUS"},
                    "clarification_status": "NEEDS_CLARIFICATION",
                },
            },
        }

        with patch("app.agent.nodes.intent_node.create_es_client") as create_es_client:
            result = await intent_node(state)

        create_es_client.assert_not_called()
        self.assertEqual("CLARIFY", result["intent_result"]["decision"])
        self.assertEqual("general_chat_subgraph", result["next_action"]["payload"]["route_target"])


if __name__ == "__main__":
    unittest.main()
