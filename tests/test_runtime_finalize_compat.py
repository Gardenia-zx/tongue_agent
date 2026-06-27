import unittest

from app.agent.runtime.graph import _finish_runtime


class RuntimeFinalizeCompatibilityTests(unittest.TestCase):
    def test_finish_runtime_merges_agent_loop_into_existing_next_action(self) -> None:
        state = {
            "next_action": {
                "type": "RESPOND_TO_USER",
                "payload": {
                    "status": "COMPLETED",
                    "answer_type": "GENERAL_CHAT",
                },
            },
            "agent_loop": {
                "iteration": 1,
                "tool_calls": [],
            },
        }

        result = _finish_runtime(
            state,
            execution_status="SUCCEEDED",
            finish_reason="final_answer",
        )

        payload = result["next_action"]["payload"]
        self.assertEqual("GENERAL_CHAT", payload["answer_type"])
        self.assertEqual("SUCCEEDED", payload["agent_loop"]["execution_status"])
        self.assertEqual("final_answer", payload["agent_loop"]["finish_reason"])

    def test_finish_runtime_builds_default_next_action(self) -> None:
        result = _finish_runtime(
            {"agent_loop": {"tool_calls": []}},
            execution_status="DEGRADED",
            finish_reason="runtime_fallback",
        )

        self.assertEqual("RESPOND_TO_USER", result["next_action"]["type"])
        self.assertEqual(
            "DEGRADED",
            result["next_action"]["payload"]["agent_loop"]["execution_status"],
        )


if __name__ == "__main__":
    unittest.main()
