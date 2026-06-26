import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.agent.nodes.intent_node import intent_node


class _FakeES:
    async def close(self) -> None:
        return None


class _FakeIntentResult:
    def model_dump(self, mode: str = "json") -> dict:
        return {
            "schema_version": "1.0",
            "engine": "TEST",
            "engine_version": "test",
            "detected_intent": "GENERAL_CHAT",
            "primary_intent": "GENERAL_CHAT",
            "secondary_intents": [],
            "confidence": 0.72,
            "decision": "ROUTE",
            "route_target": "general_chat_subgraph",
            "risk_level": "LOW",
            "missing_slots": [],
            "entities": [],
            "topk_candidates": [],
            "safety_flags": [],
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
