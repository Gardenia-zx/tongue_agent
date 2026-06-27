import asyncio
import unittest

from app.agent.turn_lifecycle import (
    begin_turn_node,
    final_response_guard_node,
    runtime_context_policy_node,
)


class TurnLifecycleTests(unittest.TestCase):
    def test_new_turn_clears_checkpointed_turn_local_state(self) -> None:
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "thread_id": "thread-1",
            "turn_guard": {"turn_id": "turn-1", "status": "PASSED"},
            "response_message": {
                "role": "assistant",
                "content": "上一轮回答",
                "turn_id": "turn-1",
            },
            "next_action": {"type": "RESPOND_TO_USER", "turn_id": "turn-1"},
            "query_context": {"turn_id": "turn-1"},
            "prompt_context": {"turn_id": "turn-1"},
            "current_turn": {"turn_id": "turn-1"},
            "agent_loop": {"turn_id": "turn-1"},
            "memory_context": {"recent_turns": ["preserved externally"]},
            "context_bundle": {"conversation_id": "conversation-1"},
        }

        result = asyncio.run(begin_turn_node(state))

        self.assertIsNone(result["response_message"])
        self.assertIsNone(result["next_action"])
        self.assertIsNone(result["query_context"])
        self.assertIsNone(result["prompt_context"])
        self.assertIsNone(result["current_turn"])
        self.assertIsNone(result["agent_loop"])
        self.assertIsNone(result["memory_context"])
        self.assertEqual("conversation-1", result["context_bundle"]["conversation_id"])
        self.assertTrue(result["turn_guard"]["reset_applied"])
        self.assertEqual("turn-2", result["turn_guard"]["turn_id"])

    def test_same_turn_resume_does_not_destroy_inflight_state(self) -> None:
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "turn_guard": {"turn_id": "turn-2", "status": "STARTED"},
            "agent_loop": {"iteration": 2},
            "response_message": {"role": "assistant", "content": "处理中"},
        }

        result = asyncio.run(begin_turn_node(state))

        self.assertEqual(2, result["agent_loop"]["iteration"])
        self.assertEqual("处理中", result["response_message"]["content"])
        self.assertFalse(result["turn_guard"]["reset_applied"])

    def test_response_guard_repairs_foreign_turn_response(self) -> None:
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "current_node": "agent_runtime.answer_finalize",
            "turn_guard": {"turn_id": "turn-2"},
            "response_message": {
                "role": "assistant",
                "content": "上一轮回答",
                "turn_id": "turn-1",
            },
            "next_action": {
                "type": "RESPOND_TO_USER",
                "turn_id": "turn-1",
                "payload": {"status": "COMPLETED"},
            },
        }

        result = asyncio.run(final_response_guard_node(state))

        self.assertEqual("REPAIRED", result["turn_guard"]["status"])
        self.assertEqual("STALE_RESPONSE_DETECTED", result["response_message"]["error_code"])
        self.assertEqual("turn-2", result["response_message"]["turn_id"])
        self.assertEqual("FAILED", result["next_action"]["payload"]["status"])

    def test_response_guard_stamps_valid_current_response(self) -> None:
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "turn_guard": {"turn_id": "turn-2"},
            "response_message": {"role": "assistant", "content": "本轮回答"},
            "next_action": {
                "type": "RESPOND_TO_USER",
                "payload": {"status": "COMPLETED"},
            },
        }

        result = asyncio.run(final_response_guard_node(state))

        self.assertEqual("PASSED", result["turn_guard"]["status"])
        self.assertEqual("turn-2", result["response_message"]["turn_id"])
        self.assertEqual("request-2", result["response_message"]["request_id"])
        self.assertEqual("turn-2", result["next_action"]["payload"]["turn_id"])

    def test_independent_query_drops_report_and_history_from_runtime(self) -> None:
        query_context = {
            "turn_id": "turn-2",
            "raw_user_input": "我最近食欲不好",
            "standalone_query": "我最近食欲不好",
            "route_hint": "health_qa_subgraph",
            "reference_resolution": {
                "target_type": "GENERAL_TOPIC",
                "is_context_dependent": False,
            },
        }
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "message": {"role": "user", "content": "我最近食欲不好"},
            "query_context": query_context,
            "intent_result": {"route_target": "health_qa_subgraph"},
            "current_turn": {
                "turn_id": "turn-2",
                "query_context": query_context,
                "query_rewrite_context": {
                    "active_report_ref": {"report_id": 10},
                    "recent_messages": [{"role": "assistant", "content": "旧回答"}],
                    "last_final_answer": {"content": "旧回答"},
                },
                "final_prompt_context": {
                    "active_report": {"report_id": 10},
                    "active_report_ref": {"report_id": 10},
                    "recent_messages": [{"role": "assistant", "content": "旧回答"}],
                    "last_final_answer": {"content": "旧回答"},
                },
            },
            "prompt_context": {
                "active_report": {"report_id": 10},
                "recent_messages": [{"role": "assistant", "content": "旧回答"}],
            },
            "context_bundle": {
                "active_report": {"report_id": 10},
                "recent_messages": [{"role": "assistant", "content": "旧回答"}],
            },
            "client_context": {
                "extra": {"latest_report": {"report_id": 10}}
            },
        }

        result = asyncio.run(runtime_context_policy_node(state))

        self.assertFalse(result["context_policy"]["allow_report_context"])
        self.assertFalse(result["context_policy"]["allow_conversation_history"])
        self.assertNotIn("active_report", result["prompt_context"])
        self.assertEqual([], result["prompt_context"]["recent_messages"])
        self.assertNotIn("active_report", result["context_bundle"])
        self.assertNotIn("latest_report", result["client_context"]["extra"])

    def test_explicit_report_followup_preserves_report_context(self) -> None:
        query_context = {
            "turn_id": "turn-2",
            "raw_user_input": "根据刚才的报告给我饮食建议",
            "standalone_query": "根据当前报告给出饮食建议",
            "route_hint": "report_followup_subgraph",
            "reference_resolution": {
                "target_type": "ACTIVE_REPORT",
                "is_context_dependent": True,
                "target_report_id": 10,
            },
        }
        state = {
            "turn_id": "turn-2",
            "request_id": "request-2",
            "message": {"role": "user", "content": "根据刚才的报告给我饮食建议"},
            "query_context": query_context,
            "current_turn": {
                "turn_id": "turn-2",
                "query_context": query_context,
                "query_rewrite_context": {"active_report_ref": {"report_id": 10}},
                "final_prompt_context": {"active_report": {"report_id": 10}},
            },
            "prompt_context": {
                "active_report": {"report_id": 10},
                "active_report_ref": {"report_id": 10},
            },
            "context_bundle": {"active_report": {"report_id": 10}},
        }

        result = asyncio.run(runtime_context_policy_node(state))

        self.assertTrue(result["context_policy"]["allow_report_context"])
        self.assertTrue(result["context_policy"]["allow_conversation_history"])
        self.assertNotIn("active_report", result["prompt_context"])
        self.assertEqual(10, result["prompt_context"]["active_report_ref"]["report_id"])


if __name__ == "__main__":
    unittest.main()
