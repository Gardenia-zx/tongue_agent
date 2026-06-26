import asyncio
import unittest

from app.agent.runtime.graph import _execute_tool_call
from app.agent.tool_policy import can_use_report_followup
from app.agent.tooling import HEALTH_QA_TOOL, REPORT_FOLLOWUP_TOOL, choose_agent_tool


class AgentToolPolicyTests(unittest.TestCase):
    def _state(self, *, text: str, route_hint: str, target_type: str, route_target: str):
        query_context = {
            "turn_id": "turn-2",
            "raw_user_input": text,
            "standalone_query": text,
            "route_hint": route_hint,
            "reference_resolution": {
                "target_type": target_type,
                "is_context_dependent": target_type != "GENERAL_TOPIC",
            },
        }
        return {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "message": {"role": "user", "content": text},
            "query_context": query_context,
            "current_turn": {
                "turn_id": "turn-2",
                "query_context": query_context,
                "query_rewrite_context": {"active_report_ref": {"report_id": 10}},
                "final_prompt_context": {"active_report": {"report_id": 10}},
            },
            "context_bundle": {"active_report": {"report_id": 10}},
            "intent_result": {"route_target": route_target},
        }

    def test_active_report_does_not_bind_unrelated_health_question(self) -> None:
        state = self._state(
            text="我最近食欲不好",
            route_hint="health_qa_subgraph",
            target_type="GENERAL_TOPIC",
            route_target="health_qa_subgraph",
        )

        tool, reason = choose_agent_tool(state)

        self.assertFalse(can_use_report_followup(state))
        self.assertEqual(HEALTH_QA_TOOL, tool.name)
        self.assertIn("health_qa", reason)

    def test_explicit_report_reference_uses_report_followup(self) -> None:
        state = self._state(
            text="根据刚才的报告给出饮食建议",
            route_hint="report_followup_subgraph",
            target_type="ACTIVE_REPORT",
            route_target="report_explanation_subgraph",
        )

        tool, reason = choose_agent_tool(state)

        self.assertTrue(can_use_report_followup(state))
        self.assertEqual(REPORT_FOLLOWUP_TOOL, tool.name)
        self.assertEqual("explicit_report_binding", reason)

    def test_runtime_rejects_report_tool_without_explicit_binding(self) -> None:
        state = self._state(
            text="我最近食欲不好",
            route_hint="health_qa_subgraph",
            target_type="GENERAL_TOPIC",
            route_target="health_qa_subgraph",
        )

        next_state, result = asyncio.run(
            _execute_tool_call(
                state=state,
                tool_name="report_context_tool",
                arguments={"reason": "read report"},
            )
        )

        self.assertIs(next_state, state)
        self.assertEqual("REJECTED", result["status"])
        self.assertEqual("explicit_report_reference_required", result["error"])
        self.assertEqual("health_qa_tool", result["recommended_tool"])


if __name__ == "__main__":
    unittest.main()
