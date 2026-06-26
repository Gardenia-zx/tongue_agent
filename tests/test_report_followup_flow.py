import unittest
from unittest.mock import AsyncMock, patch

from app.agent.nodes.query_rewrite_node import query_rewrite_node
from app.agent.nodes.report_followup_node import report_followup_node
from app.agent.nodes.route_node import route_node, select_next_route


class TestReportFollowupFlow(unittest.IsolatedAsyncioTestCase):
    def _followup_state(self) -> dict:
        return {
            "thread_id": "followup_thread",
            "user_id": 10001,
            "message": {
                "role": "user",
                "content_type": "text",
                "content": "对我的舌象分析部分，分析的详细一点",
                "attachments": [],
            },
            "client_context": {
                "page": "ai_chat",
                "extra": {
                    "latest_report": {
                        "report_id": 12,
                        "feature_summary": "本次图像识别到的主要舌象特征包括：白苔。",
                        "summary": "本次结果\n图片主要识别到白苔，结合睡眠不好和疲惫，可继续观察舌苔厚薄、润燥和近期精神状态。",
                    },
                    "recent_messages": [
                        {
                            "role": "assistant",
                            "content": "本次结果\n图片主要识别到白苔，结合睡眠不好和疲惫，可继续观察舌苔厚薄、润燥和近期精神状态。",
                            "report_id": 12,
                        }
                    ],
                },
            },
            "intent_result": {
                "primary_intent": "TONGUE_ANALYSIS_START",
                "detected_intent": "TONGUE_ANALYSIS_START",
                "route_target": "tongue_analysis_subgraph",
                "decision": "ROUTE",
                "risk_level": "LOW",
            },
        }

    def _context_bundle_followup_state(self) -> dict:
        state = self._followup_state()
        state["client_context"] = {
            "page": "ai_chat",
            "extra": {},
        }
        state["context_bundle"] = {
            "conversation_id": "90001",
            "active_report": {
                "report_id": 22,
                "feature_summary": "白苔",
                "summary": "图片主要识别到白苔，用户描述睡眠差、容易惊醒。",
            },
            "conversation_summary": {
                "summary_id": 5,
                "text": "用户上传过舌象图片，报告识别为白苔，并追问睡眠相关解释。",
            },
            "recent_messages": [
                {
                    "message_id": 101,
                    "role": "assistant",
                    "content": "图片主要识别到白苔，结合睡眠差可继续观察舌苔厚薄和精神状态。",
                    "report_id": 22,
                }
            ],
        }
        return state

    async def test_route_followup_to_report_followup_even_if_intent_is_tongue_analysis(self) -> None:
        state = self._followup_state()

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("report_followup_node", select_next_route(rewritten))
        self.assertEqual(
            "REPORT",
            rewritten["query_context"]["reference_resolution"]["target_type"],
        )
        self.assertEqual(
            "report_followup_subgraph",
            routed["next_action"]["payload"]["route_target"],
        )
        self.assertEqual(
            "report_followup_node",
            routed["next_action"]["payload"]["next_node"],
        )

    async def test_report_followup_node_uses_report_context(self) -> None:
        state = self._followup_state()
        mocked_followup = AsyncMock(
            return_value=(
                "可以，我基于前面那份报告展开说明。",
                None,
                {"answer_type": "REPORT_FOLLOWUP", "summary": "可以继续展开。"},
            )
        )

        with patch(
            "app.agent.nodes.report_followup_node.generate_report_followup_reply",
            mocked_followup,
        ):
            result = await report_followup_node(state)

        mocked_followup.assert_awaited_once()
        self.assertEqual("report_followup_node", result["current_node"])
        self.assertEqual(
            "可以，我基于前面那份报告展开说明。",
            result["response_message"]["content"],
        )
        self.assertEqual(
            "report_followup_with_context",
            result["tool_decision"]["reason"],
        )
        self.assertEqual(
            "REPORT_FOLLOWUP",
            result["response_message"]["structured_content"]["answer_type"],
        )

    async def test_report_followup_node_uses_context_bundle_active_report(self) -> None:
        state = self._context_bundle_followup_state()
        mocked_followup = AsyncMock(
            return_value=(
                "可以，我基于当前会话报告继续展开。",
                None,
                {"answer_type": "REPORT_FOLLOWUP", "summary": "继续展开。"},
            )
        )

        with patch(
            "app.agent.nodes.report_followup_node.generate_report_followup_reply",
            mocked_followup,
        ):
            result = await report_followup_node(state)

        mocked_followup.assert_awaited_once()
        self.assertEqual(
            "可以，我基于当前会话报告继续展开。",
            result["response_message"]["content"],
        )

    async def test_diet_recommendation_followup_uses_report_context(self) -> None:
        state = self._context_bundle_followup_state()
        state["message"]["content"] = "有更详细的饮食推荐吗"

        rewritten = await query_rewrite_node(state)
        routed = await route_node(rewritten)

        self.assertEqual("report_followup_node", select_next_route(rewritten))
        self.assertEqual(
            "REPORT",
            rewritten["query_context"]["reference_resolution"]["target_type"],
        )
        self.assertEqual(
            "report_followup_subgraph",
            routed["next_action"]["payload"]["route_target"],
        )

    async def test_report_followup_fetches_rag_for_diet_question(self) -> None:
        from app.agent.nodes.report_followup_node import generate_report_followup_reply

        state = self._context_bundle_followup_state()
        state["message"]["content"] = "有更详细的饮食推荐吗"
        rag_result = {
            "answer": "饮食宜清淡规律，少生冷甜腻。",
            "hits": [{"chunk_id": "c1", "content": "测试依据"}],
            "grounded": True,
        }
        llm_payload = (
            '{"content":"可以，结合当前报告和知识库，饮食上建议清淡规律，少生冷甜腻。",'
            '"structured_content":{"answer_type":"REPORT_FOLLOWUP","title":"饮食建议",'
            '"summary":"饮食上建议清淡规律，少生冷甜腻。",'
            '"sections":[{"title":"建议","items":["清淡规律","少生冷甜腻"]}],'
            '"disclaimer":"以上内容不能替代医生诊断。"}}'
        )

        with patch(
            "app.agent.nodes.report_followup_node.answer_with_rag",
            AsyncMock(return_value=rag_result),
        ) as mocked_rag, patch(
            "app.agent.nodes.report_followup_node.get_chat_model_client",
        ) as mocked_client:
            mocked_client.return_value.generate = AsyncMock(return_value=llm_payload)
            content, rag_context, structured_content = await generate_report_followup_reply(state)

        mocked_rag.assert_awaited_once()
        self.assertEqual(rag_result, rag_context)
        self.assertIn("知识库", content)
        self.assertIn("清淡规律", content)
        self.assertEqual("REPORT_FOLLOWUP", structured_content["answer_type"])
        self.assertEqual("饮食建议", structured_content["title"])

    async def test_diet_followup_fallback_still_returns_diet_structure(self) -> None:
        from app.agent.nodes.report_followup_node import generate_report_followup_reply

        state = self._context_bundle_followup_state()
        state["message"]["content"] = "有更详细的饮食推荐吗"
        rag_result = {
            "answer": "饮食宜清淡规律，少生冷甜腻，注意脾胃运化。",
            "hits": [{"chunk_id": "c1", "content": "测试依据"}],
            "grounded": True,
        }

        with patch(
            "app.agent.nodes.report_followup_node.answer_with_rag",
            AsyncMock(return_value=rag_result),
        ), patch(
            "app.agent.nodes.report_followup_node.get_chat_model_client",
        ) as mocked_client:
            mocked_client.return_value.generate = AsyncMock(return_value="不是 JSON")
            content, rag_context, structured_content = await generate_report_followup_reply(state)

        self.assertEqual(rag_result, rag_context)
        self.assertIn("饮食建议", content)
        self.assertIn("生冷", content)
        self.assertEqual("饮食建议", structured_content["title"])
        self.assertEqual("饮食建议", structured_content["sections"][0]["title"])
        self.assertNotEqual("可以重点看", structured_content["sections"][0]["title"])

    async def test_short_followup_after_diet_answer_keeps_diet_structure(self) -> None:
        from app.agent.nodes.query_rewrite_node import query_rewrite_node
        from app.agent.nodes.report_followup_node import generate_report_followup_reply

        state = self._context_bundle_followup_state()
        state["message"]["content"] = "还是不够详细"
        state["context_bundle"]["last_final_answer"] = {
            "message_id": 202,
            "role": "assistant",
            "content": "饮食建议：近期少吃生冷、油腻、甜腻，三餐规律，观察腹胀和大便状态。",
            "report_id": 22,
            "metadata": {
                "node_name": "report_followup_node",
                "route_target": "report_followup_subgraph",
                "answer_type": "REPORT_FOLLOWUP",
                "structured_content": {
                    "answer_type": "REPORT_FOLLOWUP",
                    "title": "饮食建议",
                    "sections": [{"title": "饮食建议", "items": ["少生冷甜腻"]}],
                },
            },
        }
        state["context_bundle"]["recent_messages"] = [
            state["context_bundle"]["last_final_answer"]
        ]
        rewritten = await query_rewrite_node(state)
        rag_result = {
            "answer": "饮食宜清淡规律，少生冷甜腻，注意观察腹胀、大便和食欲变化。",
            "hits": [{"chunk_id": "c1", "content": "测试依据"}],
            "grounded": True,
        }

        with patch(
            "app.agent.nodes.report_followup_node.answer_with_rag",
            AsyncMock(return_value=rag_result),
        ), patch(
            "app.agent.nodes.report_followup_node.get_chat_model_client",
        ) as mocked_client:
            mocked_client.return_value.generate = AsyncMock(return_value="不是 JSON")
            content, rag_context, structured_content = await generate_report_followup_reply(rewritten)

        self.assertEqual("DIET_ADVICE", rewritten["query_context"]["reference_resolution"]["target_focus"])
        self.assertEqual(rag_result, rag_context)
        self.assertIn("饮食建议", content)
        self.assertEqual("饮食建议", structured_content["title"])
        self.assertEqual("饮食建议", structured_content["sections"][0]["title"])
        self.assertNotEqual("可以重点看", structured_content["sections"][0]["title"])

    async def test_detailed_report_request_returns_full_report_structure(self) -> None:
        from app.agent.nodes.report_followup_node import generate_report_followup_reply

        state = self._context_bundle_followup_state()
        state["message"]["content"] = "给我一个更详细的报告"
        rag_result = {
            "answer": "白苔需要结合厚薄、润燥、腻腐以及饮食睡眠等情况综合观察。",
            "hits": [{"chunk_id": "c1", "content": "测试依据"}],
            "grounded": True,
        }

        with patch(
            "app.agent.nodes.report_followup_node.answer_with_rag",
            AsyncMock(return_value=rag_result),
        ) as mocked_rag, patch(
            "app.agent.nodes.report_followup_node.get_chat_model_client",
        ) as mocked_client:
            mocked_client.return_value.generate = AsyncMock(return_value="不是 JSON")
            content, rag_context, structured_content = await generate_report_followup_reply(state)

        mocked_rag.assert_awaited_once()
        self.assertEqual(rag_result, rag_context)
        self.assertIn("更详细的健康参考报告", content)
        self.assertEqual("DETAILED_TONGUE_REPORT", structured_content["answer_type"])
        self.assertEqual("详细舌象报告", structured_content["title"])
        section_titles = [section["title"] for section in structured_content["sections"]]
        self.assertIn("识别结果", section_titles)
        self.assertIn("舌象说明", section_titles)
        self.assertIn("健康管理建议", section_titles)
        self.assertNotIn("可以重点看", section_titles)


if __name__ == "__main__":
    unittest.main()
