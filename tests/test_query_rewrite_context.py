import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agent.nodes.query_rewrite_node import query_rewrite_node
from app.agent.nodes.route_node import route_node, select_next_route


class TestQueryRewriteContextRouting(unittest.IsolatedAsyncioTestCase):
    def _state(
        self,
        text: str,
        *,
        active_report: dict | None = None,
        last_answer: dict | None = None,
        intent_route: str = "general_chat_subgraph",
    ) -> dict:
        context_bundle = {
            "conversation_id": "ctx_001",
            "recent_messages": [],
            "last_final_answer": last_answer or {},
            "active_report": active_report,
            "conversation_summary": None,
        }
        if last_answer:
            context_bundle["recent_messages"].append(last_answer)

        return {
            "schema_version": "1.0",
            "request_id": "query_rewrite_test",
            "trace_id": "trace_query_rewrite_test",
            "user_id": 10001,
            "thread_id": "thread_query_rewrite_test",
            "conversation_id": "ctx_001",
            "message": {
                "role": "user",
                "content_type": "text",
                "content": text,
                "attachments": [],
            },
            "context_bundle": context_bundle,
            "client_context": {
                "page": "ai_chat",
                "extra": {"context_bundle": context_bundle},
            },
            "intent_result": {
                "primary_intent": "GENERAL_CHAT",
                "detected_intent": "GENERAL_CHAT",
                "route_target": intent_route,
                "decision": "ROUTE",
                "risk_level": "LOW",
            },
        }

    def _active_report(self) -> dict:
        return {
            "report_id": 6,
            "summary": "本次图片主要识别到白苔，用户描述睡眠不好、容易疲惫。",
            "feature_summary": "白苔",
            "detected_feature_codes": ["coating.color.white"],
            "rag_evidence": [],
        }

    def _assistant(
        self,
        *,
        content: str,
        node_name: str,
        answer_type: str,
        route_target: str,
        report_id: int | None = None,
    ) -> dict:
        metadata = {
            "node_name": node_name,
            "answer_type": answer_type,
            "route_target": route_target,
        }
        if report_id is not None:
            metadata["report_id"] = report_id
        return {
            "message_id": 123,
            "role": "assistant",
            "content_type": "text",
            "content": content,
            "report_id": report_id,
            "metadata": metadata,
        }

    async def test_explicit_report_followup_routes_to_report_node(self) -> None:
        state = self._state(
            "给我一个更详细的报告",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="本次图片主要识别到白苔。",
                node_name="tongue_report_node",
                answer_type="TONGUE_REPORT",
                route_target="tongue_analysis_subgraph",
                report_id=6,
            ),
            intent_route="general_chat_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("ACTIVE_REPORT", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("report_followup_subgraph", rewritten["query_context"]["route_hint"])
        self.assertIn("当前活动舌象报告", rewritten["query_context"]["standalone_query"])
        self.assertEqual("report_followup_node", select_next_route(rewritten))
        self.assertEqual("report_followup_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_health_qa_followup_uses_last_answer_even_with_active_report(self) -> None:
        state = self._state(
            "讲详细一点",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="脾虚是中医里描述脾胃运化功能偏弱的一类状态。",
                node_name="health_qa_node",
                answer_type="HEALTH_QA",
                route_target="health_qa_subgraph",
            ),
            intent_route="general_chat_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("LAST_ANSWER", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("health_qa_subgraph", rewritten["query_context"]["route_hint"])
        self.assertIn("上一轮健康问答", rewritten["query_context"]["standalone_query"])
        self.assertEqual("health_qa_node", select_next_route(rewritten))
        self.assertEqual("health_qa_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_diet_followup_after_health_qa_does_not_route_to_report(self) -> None:
        state = self._state(
            "饮食怎么注意",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="湿气重通常和体内水湿停留的健康知识解释有关。",
                node_name="health_qa_node",
                answer_type="HEALTH_QA",
                route_target="health_qa_subgraph",
            ),
            intent_route="general_chat_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("LAST_ANSWER", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("health_qa_subgraph", rewritten["query_context"]["route_hint"])
        self.assertEqual("health_qa_node", select_next_route(rewritten))
        self.assertEqual("health_qa_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_general_chat_followup_stays_general_chat(self) -> None:
        state = self._state(
            "详细点",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="我可以帮你做舌象分析、健康知识问答、报告解释和隐私请求处理。",
                node_name="general_chat_node",
                answer_type="GENERAL_CHAT",
                route_target="general_chat_subgraph",
            ),
            intent_route="health_qa_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("LAST_ANSWER", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("general_chat_subgraph", rewritten["query_context"]["route_hint"])
        self.assertEqual("general_chat_node", select_next_route(rewritten))
        self.assertEqual("general_chat_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_health_question_about_white_coating_is_not_report_followup(self) -> None:
        state = self._state(
            "白苔是什么",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="本次图片主要识别到白苔。",
                node_name="tongue_report_node",
                answer_type="TONGUE_REPORT",
                route_target="tongue_analysis_subgraph",
                report_id=6,
            ),
            intent_route="health_qa_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("GENERAL_TOPIC", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("health_qa_node", select_next_route(rewritten))
        self.assertEqual("health_qa_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_report_request_without_active_report_does_not_fabricate_context(self) -> None:
        state = self._state(
            "给我一个更详细的报告",
            active_report=None,
            last_answer=None,
            intent_route="general_chat_subgraph",
        )

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("UNKNOWN", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("general_chat_node", select_next_route(rewritten))
        self.assertEqual("general_chat_subgraph", routed["next_action"]["payload"]["route_target"])

    async def test_high_confidence_rule_does_not_call_llm(self) -> None:
        state = self._state(
            "解释一下我的舌象报告",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="本次图片主要识别到白苔。",
                node_name="tongue_report_node",
                answer_type="TONGUE_REPORT",
                route_target="tongue_analysis_subgraph",
                report_id=6,
            ),
        )

        with patch("app.agent.nodes.query_rewrite_node._generate_with_model", AsyncMock()) as mocked_llm:
            rewritten = await query_rewrite_node(state)

        mocked_llm.assert_not_awaited()
        self.assertEqual("RULE", rewritten["query_context"]["resolution_strategy"])
        self.assertEqual("ACTIVE_REPORT", rewritten["query_context"]["reference_resolution"]["target_type"])

    async def test_low_confidence_item_reference_uses_llm_fallback(self) -> None:
        state = self._state(
            "第二个再详细点",
            active_report=self._active_report(),
            last_answer=self._assistant(
                content="1. 白苔。2. 睡眠不好。",
                node_name="tongue_report_node",
                answer_type="TONGUE_REPORT",
                route_target="tongue_analysis_subgraph",
                report_id=6,
            ),
        )
        llm_payload = {
            "standalone_query": "请详细解释上一轮回答中的第二项：睡眠不好。",
            "route_hint": "report_followup_subgraph",
            "rewrite_confidence": 0.9,
            "reference_resolution": {
                "status": "RESOLVED",
                "target_type": "LAST_ANSWER_ITEM",
                "target_focus": "GENERAL_DETAIL",
                "is_context_dependent": True,
                "confidence": 0.9,
                "target_message_id": 123,
                "target_report_id": 6,
                "evidence_sources": ["raw_user_input", "last_final_answer", "active_report_ref"],
                "reason": "user refers to second item in last answer",
            },
        }

        with patch(
            "app.agent.nodes.query_rewrite_node._generate_with_model",
            AsyncMock(return_value=json.dumps(llm_payload, ensure_ascii=False)),
        ) as mocked_llm:
            rewritten = await query_rewrite_node(state)

        mocked_llm.assert_awaited_once()
        self.assertEqual("LLM_FALLBACK", rewritten["query_context"]["resolution_strategy"])
        self.assertEqual("LAST_ANSWER_ITEM", rewritten["query_context"]["reference_resolution"]["target_type"])
        self.assertEqual("report_followup_subgraph", rewritten["query_context"]["route_hint"])

    async def test_low_confidence_llm_timeout_needs_clarification(self) -> None:
        state = self._state(
            "第二个再详细点",
            last_answer=self._assistant(
                content="1. 白苔。2. 睡眠不好。",
                node_name="health_qa_node",
                answer_type="HEALTH_QA",
                route_target="health_qa_subgraph",
            ),
        )

        with patch(
            "app.agent.nodes.query_rewrite_node._generate_with_model",
            AsyncMock(side_effect=TimeoutError("llm timeout")),
        ):
            rewritten = await query_rewrite_node(state)

        self.assertEqual("NEEDS_CLARIFICATION", rewritten["query_context"]["clarification_status"])
        self.assertEqual("AMBIGUOUS", rewritten["query_context"]["reference_resolution"]["target_type"])

    async def test_llm_fallback_rejects_fabricated_message_id(self) -> None:
        state = self._state(
            "第二个再详细点",
            last_answer=self._assistant(
                content="1. 白苔。2. 睡眠不好。",
                node_name="health_qa_node",
                answer_type="HEALTH_QA",
                route_target="health_qa_subgraph",
            ),
        )
        llm_payload = {
            "standalone_query": "fabricated",
            "route_hint": "health_qa_subgraph",
            "rewrite_confidence": 0.9,
            "reference_resolution": {
                "status": "RESOLVED",
                "target_type": "LAST_ANSWER_ITEM",
                "target_focus": "GENERAL_DETAIL",
                "is_context_dependent": True,
                "confidence": 0.9,
                "target_message_id": "does-not-exist",
                "evidence_sources": ["last_final_answer"],
                "reason": "bad id",
            },
        }

        with patch(
            "app.agent.nodes.query_rewrite_node._generate_with_model",
            AsyncMock(return_value=json.dumps(llm_payload)),
        ):
            rewritten = await query_rewrite_node(state)

        self.assertEqual("NEEDS_CLARIFICATION", rewritten["query_context"]["clarification_status"])
        self.assertEqual("AMBIGUOUS", rewritten["query_context"]["reference_resolution"]["target_type"])

    async def test_stale_query_context_is_not_reused(self) -> None:
        state = self._state(
            "白苔是什么",
            active_report=self._active_report(),
        )
        state["turn_id"] = "turn-new"
        state["current_turn"] = {
            "turn_id": "turn-new",
            "query_rewrite_context": {
                "turn_id": "turn-new",
                "context_version": "query_rewrite_context.v1",
                "raw_user_input": "白苔是什么",
                "recent_messages": [],
            },
            "query_context": {
                "turn_id": "turn-new",
                "context_version": "old-context-version",
                "standalone_query": "stale",
                "reference_resolution": {"target_type": "ACTIVE_REPORT"},
            },
        }

        rewritten = await query_rewrite_node(state)

        self.assertNotEqual("stale", rewritten["query_context"]["standalone_query"])
        self.assertEqual("query_rewrite_context.v1", rewritten["query_context"]["context_version"])


if __name__ == "__main__":
    unittest.main()
