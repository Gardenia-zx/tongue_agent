import asyncio
import unittest

from app.agent.runtime.graph import (
    budget_guard_node,
    build_agent_runtime_subgraph,
    plan_validator_node,
    route_after_budget,
    route_after_validation,
)


class AgentRuntimeGraphTests(unittest.TestCase):
    def test_runtime_subgraph_compiles(self) -> None:
        compiled = build_agent_runtime_subgraph().compile()
        self.assertIsNotNone(compiled)

    def test_unknown_tool_is_rejected_before_execution(self) -> None:
        state = {
            "agent_loop": {
                "iteration": 1,
                "messages": [],
                "tool_calls": [],
                "raw_tool_calls": [
                    {
                        "id": "tool-call-1",
                        "function": {
                            "name": "dangerous_unknown_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        }

        result = asyncio.run(plan_validator_node(state))
        runtime = result["agent_loop"]

        self.assertEqual("CHECK_BUDGET", route_after_validation(result))
        self.assertEqual([], runtime["pending_tool_calls"])
        self.assertEqual("REJECTED", runtime["tool_calls"][0]["status"])
        self.assertEqual("unknown_tool", runtime["tool_calls"][0]["output"]["error"])

    def test_report_generation_requires_tongue_features(self) -> None:
        state = {
            "agent_loop": {
                "iteration": 1,
                "messages": [],
                "tool_calls": [],
                "raw_tool_calls": [
                    {
                        "id": "tool-call-2",
                        "function": {
                            "name": "tongue_report_generate_tool",
                            "arguments": '{"reason":"generate report"}',
                        },
                    }
                ],
            }
        }

        result = asyncio.run(plan_validator_node(state))
        runtime = result["agent_loop"]

        self.assertEqual([], runtime["pending_tool_calls"])
        self.assertEqual(
            "tongue_features_required",
            runtime["tool_calls"][0]["output"]["error"],
        )

    def test_budget_guard_stops_at_iteration_limit(self) -> None:
        state = {
            "agent_loop": {
                "iteration": 5,
                "max_iterations": 5,
                "max_tool_calls": 10,
                "tool_calls": [],
            }
        }

        result = asyncio.run(budget_guard_node(state))

        self.assertEqual("FALLBACK", route_after_budget(result))
        self.assertEqual(
            "max_iterations_reached",
            result["agent_loop"]["finish_reason"],
        )


if __name__ == "__main__":
    unittest.main()
