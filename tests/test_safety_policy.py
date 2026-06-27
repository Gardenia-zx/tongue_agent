import unittest

from app.agent.nodes.agent_gate_node import select_agent_gate_next
from app.intent.rules import match_safety_intent
from app.schemas.intent import RiskLevel


class SafetyPolicyTests(unittest.TestCase):
    def test_medicine_and_prescription_reference_are_allowed(self) -> None:
        for text in ["我应该吃什么中成药", "给我一个方剂参考", "这个药一天吃几片"]:
            with self.subTest(text=text):
                self.assertIsNone(match_safety_intent(text))

    def test_emergency_and_diagnosis_still_route_to_safety(self) -> None:
        emergency = match_safety_intent("我现在胸痛还喘不上气")
        self.assertIsNotNone(emergency)
        self.assertEqual(RiskLevel.EMERGENCY, emergency.risk_level)

        diagnosis = match_safety_intent("帮我判断是不是肾病")
        self.assertIsNotNone(diagnosis)
        self.assertEqual(RiskLevel.HIGH, diagnosis.risk_level)

    def test_report_diet_followup_does_not_enter_safety_gate(self) -> None:
        state = {
            "intent_result": {
                "route_target": "report_followup_subgraph",
                "risk_level": "LOW",
            },
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {
                    "route_hint": "report_followup_subgraph",
                    "standalone_query": "结合报告给我具体饮食参考",
                    "clarification_status": "NOT_NEEDED",
                },
            },
        }

        self.assertEqual("agent_loop_node", select_agent_gate_next(state))


if __name__ == "__main__":
    unittest.main()
