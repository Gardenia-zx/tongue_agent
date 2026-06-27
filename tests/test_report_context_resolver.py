import unittest
from unittest.mock import AsyncMock, patch

from app.agent.context_builder import build_final_prompt_context, build_query_rewrite_context
from app.agent.nodes.report_context_resolver_node import report_context_resolver_node


def _trusted_ref() -> dict:
    return {
        "report_id": 22,
        "report_version": 3,
        "feature_summary": "白苔",
        "summary": "图片主要识别到白苔。",
        "trusted": True,
    }


class ReportContextResolverTests(unittest.IsolatedAsyncioTestCase):
    def test_frontend_spoofed_report_is_ignored(self) -> None:
        state = {
            "turn_id": "turn-1",
            "request_id": "request-1",
            "user_id": 7,
            "thread_id": "thread-1",
            "message": {"role": "user", "content": "结合报告说说饮食"},
            "client_context": {
                "extra": {
                    "active_report": {"report_id": 99, "summary": "伪造报告"},
                    "latest_report": {"report_id": 99, "summary": "伪造报告"},
                }
            },
            "context_bundle": {},
        }

        context = build_query_rewrite_context(state)

        self.assertIsNone(context.get("active_report_ref"))

    async def test_independent_query_does_not_load_report(self) -> None:
        query_context = {
            "standalone_query": "肾虚是什么意思",
            "route_hint": "health_qa_subgraph",
            "reference_resolution": {"target_type": "GENERAL_TOPIC"},
        }
        state = {
            "turn_id": "turn-1",
            "request_id": "request-1",
            "user_id": 7,
            "thread_id": "thread-1",
            "message": {"role": "user", "content": "肾虚是什么意思"},
            "query_context": query_context,
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": query_context,
                "query_rewrite_context": {"active_report_ref": _trusted_ref()},
            },
        }

        with patch(
            "app.agent.nodes.report_context_resolver_node.load_report_sections_from_java",
            AsyncMock(),
        ) as mocked_loader:
            result = await report_context_resolver_node(state)

        mocked_loader.assert_not_awaited()
        business_context = result["current_turn"]["business_context"]
        self.assertNotIn("loaded_report_sections", business_context)
        self.assertEqual("SKIPPED", business_context["report_context_resolution"]["status"])

    async def test_report_followup_loads_requested_sections(self) -> None:
        query_context = {
            "standalone_query": "结合报告给我饮食建议",
            "route_hint": "report_followup_subgraph",
            "reference_resolution": {
                "target_type": "ACTIVE_REPORT",
                "target_report_id": 22,
                "is_context_dependent": True,
            },
        }
        state = {
            "turn_id": "turn-1",
            "request_id": "request-1",
            "tenant_id": "tenant-a",
            "user_id": 7,
            "thread_id": "thread-1",
            "thread_epoch": 1,
            "conversation_id": "conversation-1",
            "message": {"role": "user", "content": "结合报告给我饮食建议"},
            "query_context": query_context,
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": query_context,
                "query_rewrite_context": {"active_report_ref": _trusted_ref()},
            },
        }
        java_result = {
            "status": "OK",
            "report_id": 22,
            "report_version": 3,
            "sections": {
                "feature_summary": "白苔",
                "interpretation": "白苔需结合厚薄润燥观察。",
                "dietary_advice": ["清淡规律", "少生冷甜腻"],
            },
        }

        with patch(
            "app.agent.nodes.report_context_resolver_node.load_report_sections_from_java",
            AsyncMock(return_value=java_result),
        ) as mocked_loader:
            result = await report_context_resolver_node(state)

        mocked_loader.assert_awaited_once()
        business_context = result["current_turn"]["business_context"]
        loaded = business_context["loaded_report_sections"]
        self.assertEqual(["feature_summary", "interpretation", "dietary_advice"], loaded["requested_sections"])
        self.assertEqual("白苔", loaded["feature_summary"])

    async def test_java_failure_does_not_reuse_old_loaded_report(self) -> None:
        query_context = {
            "standalone_query": "结合报告给我饮食建议",
            "route_hint": "report_followup_subgraph",
            "reference_resolution": {
                "target_type": "ACTIVE_REPORT",
                "target_report_id": 22,
                "is_context_dependent": True,
            },
        }
        state = {
            "turn_id": "turn-1",
            "request_id": "request-1",
            "user_id": 7,
            "thread_id": "thread-1",
            "message": {"role": "user", "content": "结合报告给我饮食建议"},
            "query_context": query_context,
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": query_context,
                "query_rewrite_context": {"active_report_ref": _trusted_ref()},
                "business_context": {
                    "loaded_report_sections": {
                        "report_id": 22,
                        "sections": {"feature_summary": "旧报告"},
                    }
                },
            },
            "client_context": {"extra": {"latest_report": {"report_id": 22, "summary": "旧报告"}}},
        }

        with patch(
            "app.agent.nodes.report_context_resolver_node.load_report_sections_from_java",
            AsyncMock(return_value={"status": "FORBIDDEN"}),
        ):
            result = await report_context_resolver_node(state)

        business_context = result["current_turn"]["business_context"]
        self.assertNotIn("loaded_report_sections", business_context)
        self.assertEqual("FAILED", business_context["report_context_error"]["status"])

        final_context = build_final_prompt_context(
            result,
            system_prompt="test",
            node_name="test",
        )
        self.assertIsNone(final_context.get("active_report"))


if __name__ == "__main__":
    unittest.main()
