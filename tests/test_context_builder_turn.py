import unittest

from app.agent.context_builder import (
    CHECKPOINTER_SOURCE,
    LEGACY_BUNDLE_SOURCE,
    MYSQL_RECOVERY_SOURCE,
    build_final_prompt_context,
    select_short_term_context,
)
from app.agent.nodes.agent_gate_node import select_agent_gate_next
from app.agent.nodes.context_builder_node import query_rewrite_context_builder_node
from app.agent.nodes.query_rewrite_node import query_rewrite_node


class TestContextBuilderTurnIsolation(unittest.IsolatedAsyncioTestCase):
    def _base_state(self, *, turn_id: str = "turn-1") -> dict:
        return {
            "tenant_id": "tenant-a",
            "user_id": 1,
            "thread_id": "thread-a",
            "thread_epoch": 1,
            "turn_id": turn_id,
            "conversation_id": "conv-a",
            "message": {
                "role": "user",
                "content_type": "text",
                "content": "continue",
                "attachments": [],
            },
            "options": {},
        }

    async def test_new_turn_replaces_current_turn_and_same_turn_preserves(self) -> None:
        state = {
            **self._base_state(turn_id="turn-2"),
            "current_turn": {
                "turn_id": "turn-1",
                "query_context": {"standalone_query": "old"},
            },
            "context_bundle": {
                "recent_messages": [
                    {"message_id": "m1", "role": "assistant", "content": "answer"}
                ]
            },
        }

        built = await query_rewrite_context_builder_node(state)

        self.assertEqual("turn-2", built["current_turn"]["turn_id"])
        self.assertNotEqual({"standalone_query": "old"}, built["current_turn"].get("query_context"))
        first_context = built["current_turn"]["query_rewrite_context"]

        replay = await query_rewrite_context_builder_node(
            {
                **built,
                "context_bundle": {
                    "recent_messages": [
                        {
                            "message_id": "m2",
                            "role": "assistant",
                            "content": "should not overwrite same turn",
                        }
                    ]
                },
            }
        )

        self.assertEqual(first_context, replay["current_turn"]["query_rewrite_context"])

    async def test_pre_builder_does_not_write_reference_resolution(self) -> None:
        state = {
            **self._base_state(),
            "context_bundle": {
                "recent_messages": [
                    {"message_id": "m1", "role": "assistant", "content": "last answer"}
                ]
            },
        }

        built = await query_rewrite_context_builder_node(state)
        rewrite_context = built["current_turn"]["query_rewrite_context"]

        self.assertNotIn("reference_resolution", rewrite_context)
        self.assertNotIn("standalone_query", rewrite_context)

    def test_short_term_source_is_exclusive(self) -> None:
        state = {
            **self._base_state(),
            "memory_context": {
                "session": {"cache_hit": True, "source": "redis"},
                "recent_turns": [
                    {
                        "request_id": "r1",
                        "user": "checkpoint user",
                        "assistant": "checkpoint assistant",
                    }
                ],
            },
            "context_bundle": {
                "recent_messages": [
                    {
                        "message_id": "legacy-1",
                        "role": "assistant",
                        "content": "legacy assistant",
                    }
                ]
            },
        }

        selected = select_short_term_context(state)

        self.assertEqual(CHECKPOINTER_SOURCE, selected["source"])
        contents = [item["content"] for item in selected["recent_messages"]]
        self.assertIn("checkpoint assistant", contents)
        self.assertNotIn("legacy assistant", contents)

        mysql_state = {
            **self._base_state(),
            "mysql_recovery_context": {
                "recent_messages": [
                    {
                        "message_id": "mysql-1",
                        "role": "assistant",
                        "content": "mysql assistant",
                    }
                ]
            },
            "context_bundle": {
                "recent_messages": [
                    {
                        "message_id": "legacy-1",
                        "role": "assistant",
                        "content": "legacy assistant",
                    }
                ]
            },
        }

        selected_mysql = select_short_term_context(mysql_state)
        self.assertEqual(MYSQL_RECOVERY_SOURCE, selected_mysql["source"])
        self.assertEqual("mysql assistant", selected_mysql["recent_messages"][0]["content"])

        legacy_state = {
            **self._base_state(),
            "context_bundle": {
                "recent_messages": [
                    {
                        "message_id": "legacy-1",
                        "role": "assistant",
                        "content": "legacy assistant",
                    }
                ]
            },
        }
        selected_legacy = select_short_term_context(legacy_state)
        self.assertEqual(LEGACY_BUNDLE_SOURCE, selected_legacy["source"])

    def test_dedupe_merges_sources_inside_selected_source(self) -> None:
        state = {
            **self._base_state(),
            "context_bundle": {
                "recent_messages": [
                    {
                        "message_id": "same-message",
                        "role": "assistant",
                        "content": "same content",
                        "source": "java_audit",
                    },
                    {
                        "message_id": "same-message",
                        "role": "assistant",
                        "content": "same content",
                        "source": "legacy_bundle",
                    },
                ]
            },
        }

        selected = select_short_term_context(state)

        self.assertEqual(1, len(selected["recent_messages"]))
        sources = selected["recent_messages"][0]["sources"]
        self.assertEqual({"java_audit", "legacy_bundle"}, {item["source"] for item in sources})

    def test_token_trim_keeps_recent_turn_before_summary_and_long_memory(self) -> None:
        state = {
            **self._base_state(),
            "options": {"context": {"token_budget": 600}},
            "memory_context": {
                "relevant_memories": [
                    {"memory_id": "mem-1", "content": "memory " * 1200}
                ],
            },
            "context_bundle": {
                "conversation_summary": {"text": "summary " * 1200},
                "recent_turns": [
                    {
                        "request_id": "r1",
                        "user": "precise previous user",
                        "assistant": "precise previous assistant",
                    }
                ],
                "recent_messages": [
                    {
                        "role": "assistant",
                        "content": "precise previous assistant",
                    }
                ],
            },
        }

        context = build_final_prompt_context(
            state,
            system_prompt="system",
            node_name="test",
            include_long_term_memory=True,
        )

        self.assertEqual([], context["related_long_term_memories"])
        self.assertTrue(context["recent_turns"])
        self.assertIn("drop_long_term_memory", context["trim_trace"])

    async def test_context_dependent_input_without_materials_needs_clarification(self) -> None:
        state = {
            **self._base_state(),
            "message": {
                "role": "user",
                "content_type": "text",
                "content": "继续详细点",
                "attachments": [],
            },
        }

        rewritten = await query_rewrite_node(state)

        self.assertEqual(
            "NEEDS_CLARIFICATION",
            rewritten["query_context"]["clarification_status"],
        )
        self.assertEqual("MISSING_CONTEXT", rewritten["query_context"]["reference_resolution"]["status"])
        self.assertEqual("general_chat_node", select_agent_gate_next(rewritten))


if __name__ == "__main__":
    unittest.main()
