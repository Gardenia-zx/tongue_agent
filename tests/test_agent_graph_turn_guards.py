import unittest

from app.agent.graph import build_agent_graph


class AgentGraphTurnGuardTests(unittest.TestCase):
    def test_parent_graph_with_turn_guards_compiles(self) -> None:
        compiled = build_agent_graph().compile()
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
